import torch
import torch.nn as nn
import torch.nn.functional as F
from .single_stage import SingleStageNet, focal_loss, RELU_INPLACE, balanced_mask_loss_ce
from .slrnet import FactorizationReconstruction, DictionaryInitialization, affine_transform, l2norm, max_onehot


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


class Net(SingleStageNet):
    def __init__(self, *args, scale_factor=0.3, use_flip=False, num_slots=64, num_iters=1, temperature=1.0,
                 pos_weight=1.0, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_factor = scale_factor
        self.use_flip = use_flip
        self.num_slots = num_slots
        self.pos_weight = pos_weight
        if class_weights is not None:
            print('apply weighted classification loss')
            class_weights = torch.Tensor(class_weights).float()
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
        self.factorization_reconstruction = FactorizationReconstruction(256, num_iters=num_iters,
                                                                        scale=1.0 / temperature)
        self.initialize_dictionary = DictionaryInitialization()
        # redefine refine
        # self._aff = RefineModule2()
        self.aux_head = nn.Sequential(nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
                                      self.norm_layer(256),
                                      nn.ReLU(RELU_INPLACE),
                                      nn.Dropout(0.5),
                                      nn.Conv2d(256, self.num_classes - 1, 1, stride=1))

        self.newly_added.append(self.factorization_reconstruction)
        self.newly_added.append(self.initialize_dictionary)
        self.newly_added.append(self.aux_head)
        self.init_weights()

    def backbone_forward(self, x):
        x = self.forward_backbone(x)
        x = self.aspp(x)
        x2_x = self.fc8_skip(self._backbone['conv3'])
        x_up = F.interpolate(x, size=x2_x.shape[-2:], mode='bilinear', align_corners=True)
        x = self.fc8_x(torch.cat([x_up, x2_x], dim=1))
        return x, self._backbone['conv3']

    def segmentation_forward(self, x, x_s):
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
        cls_2 = focal_loss(masks_.mean(-1), p=3, c=0.01)
        cls = cls_1[:, 1:] + cls_2[:, 1:]
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
            x_, _, _ = self.factorization_reconstruction.forward(x.view(B, x.size(1), -1).permute(0, 2, 1),
                                                                 dict_init.permute(0, 2, 1))
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
        x_multi_view, multi_view_coding, dictionary = self.factorization_reconstruction.forward(
            x_multi_view.permute(0, 2, 1), dict_init)
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
        target_size = x.shape[-2:]
        coding2 = inv_transform(coding2, target_size=target_size)
        x2_aug = inv_transform(x2_aug, target_size=target_size)
        mask2_aug = inv_transform(mask2_aug, target_size=target_size)

        # multi-scale test
        if test_mode:
            return (cls + cls_affine) / 2.0, \
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

        if self.class_weights is not None:
            pos_weights = self.class_weights * self.pos_weight
        else:
            pos_weights = torch.ones_like(labels[0]) * self.pos_weight

        cls_loss1 = F.binary_cross_entropy_with_logits(cls, labels, pos_weight=pos_weights)
        cls_loss2 = F.binary_cross_entropy_with_logits(cls_affine, labels, pos_weight=pos_weights)
        cls_loss_aux = F.binary_cross_entropy_with_logits(cls_aux, labels, pos_weight=pos_weights)
        cls_loss = cls_loss1 + cls_loss2 + 0.4 * cls_loss_aux

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

        return cls_loss.unsqueeze(0), seg_loss, reg_loss.unsqueeze(0), code_reg_loss.unsqueeze(0), \
               {'cam': masks, 'dec': masks_refined, 'pseudo': pseudo_gt}, dictionary
