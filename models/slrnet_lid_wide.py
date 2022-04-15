import torch
import torch.nn as nn
import torch.nn.functional as F
from .single_stage import focal_loss, RELU_INPLACE, balanced_mask_loss_ce
import math
from models.backbones.resnet38d import resnet38d
# modules
from models.mods import PAMR as RefineModule
from models.mods import StochasticGate
from models.mods import GCI

from models.mods.transforms import resize_as, resize_to, random_hflip



BN_EPS = 1e-5
BN_MOMENTUM = 0.05


class _BatchNorm(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super(_BatchNorm, self).__init__(*args, **kwargs, momentum=BN_MOMENTUM, eps=BN_EPS)


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, norm_layer):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = norm_layer(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)


class GCI(nn.Module):
    """Global Cue Injection
    Takes shallow features with low receptive
    field and augments it with global info via
    adaptive instance normalisation"""

    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(GCI, self).__init__()

        self.norm_layer = norm_layer
        self.from_scratch_layers = []

        self._init_params()

    def _conv2d(self, *args, **kwargs):
        conv = nn.Conv2d(*args, **kwargs)
        self.from_scratch_layers.append(conv)
        torch.nn.init.kaiming_normal_(conv.weight)
        return conv

    def _bnorm(self, *args, **kwargs):
        bn = self.norm_layer(*args, **kwargs)
        #self.bn_learn.append(bn)
        self.from_scratch_layers.append(bn)
        if not bn.weight is None:
            bn.weight.data.fill_(1)
            bn.bias.data.zero_()
        return bn

    def _init_params(self):

        self.fc_deep = nn.Sequential(self._conv2d(512, 1024, 1, bias=False),
                                     self._bnorm(1024), nn.ReLU(inplace=RELU_INPLACE))

        self.fc_skip = nn.Sequential(self._conv2d(512, 512, 1, bias=False),
                                     self._bnorm(512, affine=False))

        self.fc_cls = nn.Sequential(self._conv2d(512, 512, 1, bias=False),
                                    self._bnorm(512),
                                    nn.ReLU(inplace=RELU_INPLACE))

    def forward(self, x, y):
        """Forward pass

        Args:
            x: shalow features
            y: deep features
        """

        # extract global attributes
        y = self.fc_deep(y)
        attrs, _ = y.view(y.size(0), y.size(1), -1).max(-1)

        # pre-process shallow features
        x = self.fc_skip(x)
        x = F.relu(self._adin_conv(x, attrs))

        return self.fc_cls(x)

    def _adin_conv(self, x, y):

        bs, num_c, _, _ = x.size()
        assert 2*num_c == y.size(1), "AdIN: dimension mismatch"

        y = y.view(bs, 2, num_c)
        gamma, beta = y[:, 0], y[:, 1]

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return x * (gamma + 1) + beta

class ASPP(nn.Module):
    def __init__(self, inplanes, output_stride, norm_layer):
        super(ASPP, self).__init__()

        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.norm_layer= norm_layer
        self.aspp1 = _ASPPModule(inplanes, 512, 1, padding=0, dilation=dilations[0], norm_layer=norm_layer)
        self.aspp2 = _ASPPModule(inplanes, 512, 3, padding=dilations[1], dilation=dilations[1], norm_layer=norm_layer)
        self.aspp3 = _ASPPModule(inplanes, 512, 3, padding=dilations[2], dilation=dilations[2], norm_layer=norm_layer)
        self.aspp4 = _ASPPModule(inplanes, 512, 3, padding=dilations[3], dilation=dilations[3], norm_layer=norm_layer)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 512, 1, stride=1, bias=False),
                                             norm_layer(512),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(512*5, 512, 1, bias=False)
        self.bn1 = norm_layer(512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, self.norm_layer):
                if not m.weight is None:
                    m.weight.data.fill_(1)
                else:
                    print("ASPP has not weight: ", m)

                if not m.bias is None:
                    m.bias.data.zero_()
                else:
                    print("ASPP has not bias: ", m)



class SingleStageNet(nn.Module):
    def __init__(self, pretrained=True, num_classes=21, backbone='resnet50', pamr_iter=10,
                 pamr_dilations=(1, 2, 4, 8, 12, 24), cutoff_top=0.6, cutoff_low=0.2,
                 norm_layer=_BatchNorm):
        super().__init__()
        self.num_classes = num_classes
        self.norm_layer = norm_layer
        if backbone == 'resnet50':
            self.backbone = resnet50()
        elif backbone == 'resnet101':
            self.backbone = resnet101()
        elif backbone == 'resnest50':
            self.backbone = resnest50(pretrained=pretrained)
        elif backbone == 'resnet38':
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

        self.fc8_skip = nn.Sequential(nn.Conv2d(256, 96, 1, bias=False),
                                      norm_layer(96),
                                      nn.ReLU(RELU_INPLACE))
        self.fc8_x = nn.Sequential(nn.Conv2d(512+96, 512, kernel_size=3, stride=1, padding=1, bias=False),
                                   norm_layer(512),
                                   nn.ReLU(RELU_INPLACE))
        self.last_conv = nn.Sequential(nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=False),
                                       norm_layer(512),
                                       nn.ReLU(RELU_INPLACE),
                                       nn.Dropout(0.3),
                                       nn.Conv2d(512, num_classes - 1, 1, stride=1))

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



def pseudo_gtmask(mask, cutoff_top=0.5, cutoff_low=0.1, eps=1e-8):
    """Convert continuous mask into binary mask"""
    bs, c, h, w = mask.size()
    # norm masks
    mask /= F.adaptive_max_pool2d(mask, (1, 1)) + 1e-6
    mask = mask.view(bs, c, -1)

    # mask_max, _ = mask.max(-1, keepdim=True)
    # mask_max *= cutoff_top

    fg_masks = (mask[:, 1:] > cutoff_top).type_as(mask)

    # bg_mask = torch.sum(fg_masks, dim=1, keepdim=True) < 1.
    bg_mask = torch.sum((mask[:, 1:] > cutoff_low).type_as(mask), dim=1, keepdim=True) < 1
    bg_mask = bg_mask.type_as(mask)

    pseudo_gt = torch.cat([bg_mask, fg_masks], dim=1)

    # remove ambiguous pixels (sum != 1, when sum==0, no need to process)
    ambiguous = (pseudo_gt.sum(1, keepdim=True) > 1).type_as(mask)
    pseudo_gt = (1 - ambiguous) * pseudo_gt

    return pseudo_gt.view(bs, c, h, w)


def affine_transform(x, scale_factor=0.5, flip=False):
    B, _, H, W = x.shape
    out = resize_to(x, size=(int(H * scale_factor), int(W * scale_factor)))
    if flip:
        out, flip_p = random_hflip(out, return_p=True)

    def inverse_transform(y, anchor_size):
        if flip:
            y = random_hflip(y, p=flip_p, return_p=False)
        y = resize_to(y, size=anchor_size)
        return y
    return out, inverse_transform


def l2norm(inp, dim):
    '''Normlize the inp tensor with l2-norm.
    Returns a tensor where each sub-tensor of input along the given dim is
    normalized such that the 2-norm of the sub-tensor is equal to 1.
    Arguments:
        inp (tensor): The input tensor.
        dim (int): The dimension to slice over to get the ssub-tensors.
    Returns:
        (tensor) The normalized tensor.
    '''
    return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))


class DictionaryInitialization(nn.Module):
    def __init__(self):
        super(DictionaryInitialization, self).__init__()

    def forward(self, x, p):
        b, c, h, w = x.size()
        k = p.size(1)
        x = x.view(b, c, -1)
        # here scale using 1
        p = p.view(b, k, -1)                        # (b, c, n)
        p = p.softmax(dim=1)                       # (b, k, n)
        p = p / (p.sum(dim=-1, keepdim=True) + 1e-6)
        dict_ = x.matmul(p.permute(0, 2, 1))      # (b, c, n)(b, n, k)->(b, c, k)
        return dict_


class FactorizationReconstruction(nn.Module):
    def __init__(self, c, num_iters=1, scale=1.0, eps=1e-6):
        super().__init__()
        self.num_iters = num_iters

        self.transform1 = nn.Linear(c, c)
        self.transform2 = nn.Linear(c, c)

        self.scale = scale
        self.eps = eps

    def forward(self, x, inits):
        idn = x
        x = self.transform1(x)
        B, N, C = x.shape
        dictionary = self.transform1(inits)
        # x = l2norm(x, dim=-1)
        dictionary = l2norm(dictionary, dim=-1)
        coding = None
        for i in range(self.num_iters):
            dots = torch.einsum('bid,bjd->bij', dictionary, x) * self.scale  # (B, K, D) (B, N, D) -> (B, K, N)
            coding = dots.softmax(dim=1)
            attn = coding + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            dictionary = l2norm(torch.einsum('bjd,bij->bid', x, attn), dim=-1)  # (B, N, D) (B, K, N) -> (B, K, D)
        # reconstruction
        x = torch.einsum('bij,bid->bjd', coding, dictionary)  # (B, K, N)(B, K, D) -> (B, N, D)
        x = F.relu(x)

        x = self.transform2(x)
        x = idn + x
        x = F.relu(x)
        return x, coding, dictionary


class Net(SingleStageNet):
    def __init__(self, *args, focal_p=3, focal_c=0.01, scale_factor=0.3, use_flip=False, num_slots=64, num_iters=1, temperature=1.0, pos_weight=1.0, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_factor = scale_factor
        self.use_flip = use_flip
        self.num_slots = num_slots
        self.pos_weight = pos_weight
        self.focal_p = focal_p
        self.focal_c = focal_c
        if class_weights is not None:
            class_weights = torch.Tensor(class_weights).float()
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
        self.factorization_reconstruction = FactorizationReconstruction(512, num_iters=num_iters, scale=1.0 / temperature)
        self.initialize_dictionary = DictionaryInitialization()

        self.aux_head = nn.Sequential(nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=False),
                                      self.norm_layer(512),
                                      nn.ReLU(RELU_INPLACE),
                                      nn.Dropout(0.5),
                                      nn.Conv2d(512, self.num_classes - 1, 1, stride=1))
        self.fc8_up = nn.Sequential(nn.Conv2d(256, 512, 1, bias=False), self.norm_layer(512),
                                      nn.ReLU(RELU_INPLACE))

        self.newly_added.append(self.factorization_reconstruction)
        self.newly_added.append(self.initialize_dictionary)
        self.newly_added.append(self.aux_head)
        self.newly_added.append(self.fc8_up)
        self.init_weights()

    def backbone_forward(self, x):
        x = self.forward_backbone(x)
        x = self.aspp(x)
        x2_x = self.fc8_skip(self._backbone['conv3'])
        x_up = F.interpolate(x, size=x2_x.shape[-2:], mode='bilinear', align_corners=True)
        x = self.fc8_x(torch.cat([x_up, x2_x], dim=1))
        return x, self._backbone['conv3']

    def segmentation_forward(self, x, x_s):
        x_s = self.fc8_up(x_s)
        x2 = self.shallow_mask(x_s, x)
        x = self.sg(x, x2, alpha_rate=0.3)
        x = self.last_conv(x)
        bg = torch.ones_like(x[:, :1])
        x = torch.cat([bg, x], dim=1)
        masks = F.softmax(x, dim=1)
        return x, masks

    def aux_forward(self, x):
        x = self.aux_head(x)
        bg = 1 - torch.max(x, dim=1, keepdim=True)[0]
        x = torch.cat([bg, x], dim=1)
        masks = F.softmax(x, dim=1)
        return x, masks

    def classifier_forward(self, x, masks):
        """
        Pooling & focal loss
        :param x:
        :param masks:
        :return:
        """
        bs, c, h, w = x.size()
        features = x.view(bs, c, -1)
        masks_ = masks.view(bs, c, -1)
        cls_1 = (features * masks_).sum(-1) / (1.0 + masks_.sum(-1))
        # focal penalty loss
        cls_2 = focal_loss(masks_.mean(-1), p=self.focal_p, c=self.focal_c)

        cls = cls_1[:, 1:] + cls_2[:, 1:]
        # cls = F.adaptive_avg_pool2d(x, (1,1)).view(bs, c)[:, 1:]
        return cls

    def forward(self, x, x_raw=None, labels=None, single_scale=False):
        test_mode = x_raw is None and labels is None
        if not isinstance(x, torch.Tensor):
            x, x2 = x
        else:
            x2 = None
        B, _, H, W = x.shape
        if test_mode and single_scale:
            x, x_s = self.backbone_forward(x)
            x_aux, mask_aux = self.aux_forward(x)
            dict_init = self.initialize_dictionary(x, x_aux)
            x_, _, _ = self.factorization_reconstruction.forward(x.view(B, x.size(1), -1).permute(0, 2, 1), dict_init.permute(0, 2, 1))
            x = x_.permute(0, 2, 1).view_as(x)
            x, masks = self.segmentation_forward(x, x_s)
            cls = self.classifier_forward(x, masks)
            masks = F.interpolate(masks, size=(H, W), mode='bilinear', align_corners=True)
            return cls, masks

        # affine transform
        if x2 is None:
            x2, inv_transform = affine_transform(x, scale_factor=self.scale_factor, flip=self.use_flip)
        else:
            x2, inv_transform = affine_transform(x2, scale_factor=self.scale_factor, flip=self.use_flip)
        # base forward
        x, x_s = self.backbone_forward(x)
        x2, x2_s = self.backbone_forward(x2)

        # aux forward
        x_aux, mask_aux = self.aux_forward(x)
        dict_init = self.initialize_dictionary(x, x_aux.detach()).detach()
        dict_init = dict_init.permute(0, 2, 1)

        # cross-view MF
        C = x.size(1)
        x_multi_view = torch.cat([x.view(B, C, -1), x2.view(B, C, -1)], dim=-1)
        # x_multi_view: (B,N,C), multi_view_coding: (B,K,N), dictionary: (1,K,C)
        x_multi_view, multi_view_coding, dictionary = self.factorization_reconstruction.forward(x_multi_view.permute(0, 2, 1), dict_init)
        x_multi_view = x_multi_view.permute(0, 2, 1)  # (B, C, N1+N2)
        x_aug, x2_aug = torch.split(x_multi_view, (x.size(-1) * x.size(-2), x2.size(-1) * x2.size(-2)), dim=-1)
        x_aug = x_aug.reshape_as(x)
        x2_aug = x2_aug.reshape_as(x2)
        coding, coding2 = torch.split(multi_view_coding,
                                      (x.size(-1) * x.size(-2), x2.size(-1) * x2.size(-2)), dim=-1)
        coding = coding.view(coding.size(0), coding.size(1), x.size(2), x.size(3))
        coding2 = coding2.view(coding.size(0), coding.size(1), x2.size(2), x2.size(3))

        # seg forward (x2)
        x_aug, mask_aug = self.segmentation_forward(x_aug, x_s)
        x2_aug, mask2_aug = self.segmentation_forward(x2_aug, x2_s)

        # cls forward (x2)
        cls = self.classifier_forward(x_aug, mask_aug)
        cls_affine = self.classifier_forward(x2_aug, mask2_aug)

        # inverse transform
        anchor_size = x.shape[-2:]
        coding2 = inv_transform(coding2, anchor_size=anchor_size)
        x2_aug = inv_transform(x2_aug, anchor_size=anchor_size)
        mask2_aug = inv_transform(mask2_aug, anchor_size=anchor_size)

        # multi-scale test
        if test_mode:
            return (cls + cls_affine) / 2.0,\
                   F.interpolate((mask_aug + mask2_aug) / 2.0, size=(H, W),
                                 mode='bilinear', align_corners=True)

        # aux forward
        cls_aux = self.classifier_forward(x_aux, mask_aux)

        ######################## Compute losses #############################
        # loss for masks
        # bg is ignored
        reg_loss = torch.abs(mask_aug[:, 1:, :, :] - mask2_aug[:, 1:, :, :])
        reg_loss *= labels[:, :, None, None].type_as(reg_loss)
        reg_loss = torch.mean(reg_loss)

        code_reg_loss = torch.abs(max_onehot(coding) - max_onehot(coding2))
        code_reg_loss = torch.mean(code_reg_loss)

        # Ensemble & Refine ! NO NEED gradients
        with torch.no_grad():
            # masks = (mask.detach() + mask_affine.detach() + mask_aug.detach() + mask_affine_aug.detach()) / 4.0
            masks = (mask_aug.detach() + mask2_aug.detach()) / 2.0
            masks_refined = self.run_pamr(x_raw, masks)
            # rescale & clean invalid categories
            masks = F.interpolate(masks, size=(H, W), mode='bilinear', align_corners=True)
            masks_refined = F.interpolate(masks_refined, size=(H, W), mode='bilinear', align_corners=True)
            masks[:, 1:] *= labels[:, :, None, None].type_as(masks)
            masks_refined[:, 1:] *= labels[:, :, None, None].type_as(masks_refined)
            pseudo_gt = pseudo_gtmask(masks_refined, cutoff_top=self.cutoff_top, cutoff_low=self.cutoff_low).detach()
        seg_loss1 = balanced_mask_loss_ce(x_aug, pseudo_gt, labels)
        seg_loss2 = balanced_mask_loss_ce(x2_aug, pseudo_gt, labels)
        seg_loss_aux = balanced_mask_loss_ce(x_aux, pseudo_gt, labels)

        seg_loss = seg_loss1 + seg_loss2 + 0.4 * seg_loss_aux

        # classification loss
        if self.class_weights is not None:
            pos_weights = self.class_weights * self.pos_weight
        else:
            pos_weights = torch.ones_like(labels[0]) * self.pos_weight

        cls_loss1 = F.binary_cross_entropy_with_logits(cls, labels, pos_weight=pos_weights)
        cls_loss2 = F.binary_cross_entropy_with_logits(cls_affine, labels, pos_weight=pos_weights)
        cls_loss_aux = F.binary_cross_entropy_with_logits(cls_aux, labels, pos_weight=pos_weights)

        cls_loss = cls_loss1 + cls_loss2 + cls_loss_aux

        return cls_loss.unsqueeze(0), seg_loss, reg_loss.unsqueeze(0), code_reg_loss.unsqueeze(0),\
               {'cam': masks, 'dec': masks_refined, 'pseudo': pseudo_gt}, dictionary



def max_onehot(x):
    x_max = torch.max(x, dim=1, keepdim=True)[0]
    x_ = x.clone()
    x_[x != x_max] = 0.0
    return x_

