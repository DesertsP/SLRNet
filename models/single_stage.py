import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# backbone nets
from models.backbones.resnet38d import resnet38d
# modules
from models.mods import ASPP
from models.mods import PAMR as RefineModule
from models.mods import StochasticGate
from models.mods import GCI
# from models.mods.refine import RefineModule
import logging

BN_EPS = 1e-5
BN_MOMENTUM = 0.05
RELU_INPLACE = False


class _BatchNorm(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super(_BatchNorm, self).__init__(*args, **kwargs, momentum=BN_MOMENTUM, eps=BN_EPS)


class SingleStageNet(nn.Module):
    def __init__(self, pretrained=True, num_classes=21, backbone='resnet38', pamr_iter=10,
                 pamr_dilations=(1, 2, 4, 8, 12, 24), cutoff_top=0.6, cutoff_low=0.2,
                 norm_layer=_BatchNorm):
        super().__init__()
        self.num_classes = num_classes
        self.norm_layer = norm_layer
        if backbone == 'resnet38':
            self.backbone = resnet38d(pretrained=pretrained)
        else:
            raise NotImplementedError
        self.cutoff_top = cutoff_top
        self.cutoff_low = cutoff_low
        self.aspp = ASPP(self.backbone.fan_out(), 8, norm_layer)
        # self._aff = PAMR(pamr_iter, pamr_dilations)
        self._aff = RefineModule(pamr_iter, pamr_dilations)

        self.shallow_mask = GCI(norm_layer=norm_layer)
        self.sg = StochasticGate()

        self.fc8_skip = nn.Sequential(nn.Conv2d(256, 48, 1, bias=False),
                                      norm_layer(48),
                                      nn.ReLU(RELU_INPLACE))
        self.fc8_x = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                   norm_layer(256),
                                   nn.ReLU(RELU_INPLACE))
        self.last_conv = nn.Sequential(nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
                                       norm_layer(256),
                                       nn.ReLU(RELU_INPLACE),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
                                       norm_layer(256),
                                       nn.ReLU(RELU_INPLACE),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes - 1, 1, stride=1))

        self.newly_added = nn.ModuleList([self.aspp, self.shallow_mask, self.sg,
                                          self.fc8_skip, self.fc8_x, self.last_conv])
        self.init_weights()
        self.bn_frozen_layers = []
        self.fixed_layers = []
        if isinstance(self.backbone.fixed_layers, nn.Module):
            self.fixed_layers.append(self.backbone.fixed_layers)
        else:
            self.fixed_layers.extend(self.backbone.fixed_layers)
        self._fix_running_stats(self.backbone, fix_params=True)
        self._fix_running_stats(self.aspp, fix_params=False)

    def train(self, mode=True):
        super().train(mode)
        for l in self.fixed_layers:
            for p in l.parameters():
                p.requires_grad = False
        for bn_layer in self.bn_frozen_layers:
            bn_layer.eval()

    def _fix_running_stats(self, layer, fix_params=False):
        if isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            self.bn_frozen_layers.append(layer)
            if fix_params and not layer in self.fixed_layers:
                self.fixed_layers.append(layer)
        elif isinstance(layer, list):
            for m in layer:
                self._fix_running_stats(m, fix_params)
        else:
            for m in layer.children():
                self._fix_running_stats(m, fix_params)

    def parameter_groups(self):
        assert len(list(self.parameters())) == len(list(self.backbone.parameters())) \
               + len(list(self.newly_added.parameters())), 'param list error'
        groups = ([], [], [], [])
        for name, p in self.named_parameters():
            if name.startswith('backbone.') and name.endswith('.weight'):
                groups[0].append(p)
            elif name.startswith('backbone.') and name.endswith('.bias'):
                groups[1].append(p)
            elif name.endswith('.weight'):
                groups[2].append(p)
            elif name.endswith('.bias'):
                groups[3].append(p)
            elif name.endswith('.gamma') or name.endswith('mu') or name.endswith('sigma'):
                groups[2].append(p)
            else:
                groups[2].append(p)
                # logging.warning(f'=>Not support parameter: {name}')
        assert len(list(self.parameters())) == sum([len(g) for g in groups])
        return groups

    def init_weights(self):
        for m in self.newly_added.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, self.norm_layer) and m.weight is not None:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def run_pamr(self, im, mask):
        im = F.interpolate(im, mask.size()[-2:], mode="bilinear", align_corners=True)
        masks_dec = self._aff(im, mask)
        return masks_dec

    def forward_backbone(self, x):
        self._backbone = self.backbone.forward_as_dict(x)
        return self._backbone['conv6']

    def forward(self, x, x_raw=None, labels=None):
        test_mode = x_raw is None and labels is None
        x_ori = x
        x = self.forward_backbone(x)
        x = self.aspp(x)
        x2_x = self.fc8_skip(self._backbone['conv3'])
        x_up = F.interpolate(x, size=x2_x.shape[-2:], mode='bilinear', align_corners=True)
        x = self.fc8_x(torch.cat([x_up, x2_x], dim=1))
        x2 = self.shallow_mask(self._backbone['conv3'], x)
        x = self.sg(x, x2, alpha_rate=0.3)
        x = self.last_conv(x)

        bg = torch.ones_like(x[:, :1])
        x = torch.cat([bg, x], dim=1)
        bs, c, h, w = x.size()
        masks = F.softmax(x, dim=1)
        features = x.view(bs, c, -1)
        masks_ = masks.view(bs, c, -1)
        cls_1 = (features * masks_).sum(-1) / (1.0 + masks_.sum(-1))
        # focal penalty loss
        cls_2 = focal_loss(masks_.mean(-1), p=3, c=0.01)
        cls = cls_1[:, 1:] + cls_2[:, 1:]
        if test_mode:
            return cls, F.interpolate(masks, x_ori.shape[-2:], mode='bilinear', align_corners=True)

        # foreground stats
        masks_ = masks_[:, 1:]
        cls_fg = (masks_.mean(-1) * labels).sum(-1) / labels.sum(-1)

        # mask refinement with PAMR
        masks_dec = self.run_pamr(x_raw, masks.detach())
        masks = F.interpolate(masks, x_ori.shape[-2:], mode='bilinear', align_corners=True)
        masks_dec = F.interpolate(masks_dec, x_ori.shape[-2:], mode='bilinear', align_corners=True)
        masks[:, 1:] *= labels[:, :, None, None].type_as(masks)
        masks_dec[:, 1:] *= labels[:, :, None, None].type_as(masks_dec)
        pseudo_gt = pseudo_gtmask(masks_dec, cutoff_top=self.cutoff_top, cutoff_low=self.cutoff_low).detach()
        loss_mask = balanced_mask_loss_ce(x, pseudo_gt, labels)
        return cls, cls_fg, {'cam': masks, 'dec': masks_dec, 'pseudo': pseudo_gt}, x, loss_mask


def focal_loss(x, p=1, c=0.1):
    return torch.pow(1 - x, p) * torch.log(c + x)


def pseudo_gtmask(mask, cutoff_top=0.6, cutoff_low=0.2, eps=1e-8):
    """Convert continuous mask into binary mask"""
    bs, c, h, w = mask.size()
    mask = mask.view(bs, c, -1)

    # for each class extract the max confidence, 取最大值的70%，如果0.7max < 0.2，截断（类似于max norm）
    mask_max, _ = mask.max(-1, keepdim=True)
    mask_max[:, :1] *= 0.7
    mask_max[:, 1:] *= cutoff_top
    # mask_max *= cutoff_top

    # if the top score is too low, ignore it
    lowest = torch.Tensor([cutoff_low]).type_as(mask_max)
    mask_max = mask_max.max(lowest)

    pseudo_gt = (mask > mask_max).type_as(mask)

    # remove ambiguous pixels
    ambiguous = (pseudo_gt.sum(1, keepdim=True) > 1).type_as(mask)
    pseudo_gt = (1 - ambiguous) * pseudo_gt

    return pseudo_gt.view(bs, c, h, w)


def balanced_mask_loss_ce(mask, pseudo_gt, gt_labels, ignore_index=255):
    """Class-balanced CE loss
    - cancel loss if only one class in pseudo_gt
    - weight loss equally between classes
    """

    mask = F.interpolate(mask, size=pseudo_gt.size()[-2:], mode="bilinear", align_corners=True)

    # indices of the max classes
    mask_gt = torch.argmax(pseudo_gt, 1)

    # for each pixel there should be at least one 1
    # otherwise, ignore
    ignore_mask = pseudo_gt.sum(1) < 1.
    mask_gt[ignore_mask] = ignore_index

    # class weight balances the loss w.r.t. number of pixels
    # because we are equally interested in all classes
    bs, c, h, w = pseudo_gt.size()
    num_pixels_per_class = pseudo_gt.view(bs, c, -1).sum(-1)
    num_pixels_total = num_pixels_per_class.sum(-1, keepdim=True)
    class_weight = (num_pixels_total - num_pixels_per_class) / (1 + num_pixels_total)
    class_weight = (pseudo_gt * class_weight[:, :, None, None]).sum(1).view(bs, -1)

    # BCE loss
    loss = F.cross_entropy(mask, mask_gt, ignore_index=ignore_index, reduction="none")
    loss = loss.view(bs, -1)

    # we will have the loss only for batch indices
    # which have all classes in pseudo mask
    gt_num_labels = gt_labels.sum(-1).type_as(loss) + 1  # + BG
    ps_num_labels = (num_pixels_per_class > 0).type_as(loss).sum(-1)
    batch_weight = (gt_num_labels == ps_num_labels).type_as(loss)

    loss = batch_weight * (class_weight * loss).mean(-1)
    return loss


if __name__ == '__main__':
    net = SingleStageNet('pretrained_models/resnet38_ilsvrc-cls_rna-a1_cls1000_ep-0001.pth', backbone='resnet38')