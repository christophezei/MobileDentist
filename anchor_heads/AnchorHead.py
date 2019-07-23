from __future__ import division

import numpy as np
import torch.nn as nn
from mmcv.cnn import normal_init
from cores.anchor.AnchorGenerator import AnchorGenerator
from cores.anchor.anchor_target_maxiou_pseudo import anchor_target
from cores.misc import multi_apply


class AnchorHead(nn.Module):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).

    Args:
        num_classes (int): num_classes.
        feat_channels (int): Number of channels of the feature map.
        anchor_scales (Iterable): Anchor scales.
        anchor_ratios (Iterable): Anchor aspect ratios.
        anchor_strides (Iterable): Anchor strides, how often to sample.
        anchor_base_sizes (Iterable): Anchor base sizes of different resolutions.
        target_means (Iterable): Mean values of regression targets.
        target_stds (Iterable): Std values of regression targets.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
    """

    def __init__(
            self,
            num_classes,
            feat_channels,
            anchor_scales,
            anchor_ratios,
            anchor_strides,
            anchor_base_sizes,
            target_means,
            target_stds,
            sampling=False,
    ):
        super(AnchorHead, self).__init__()

        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes = anchor_base_sizes
        self.target_means = target_means
        self.target_stds = target_stds
        self.loss_cls = loss_cls
        self.loss_bbox = loss_bbox
        self.sampling = sampling

        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            # generate anchors for each base of one location.
            self.anchor_generators.append(
                AnchorGenerator(anchor_base, anchor_scales, anchor_ratios))

        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)

        # define regression operations
        self.conv_cls = nn.Conv2d(self.feat_channels, self.num_anchors * self.num_classes, 1)
        self.conv_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        normal_init(self.conv_cls, std=0.01)
        normal_init(self.conv_reg, std=0.01)

    def forward_single(self, x):
        # x: B x C x H x W
        # cls_score: B x class x H x W
        # bbox_pred: B x 4 x H x W
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        return cls_score, bbox_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, img_metas):
        """Get anchors according to feature map sizes.
        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: anchors of each image, valid flags of each image.
            anchor_list: B x N_of_Resolutions x Anchor_per_location*N_of_locations x 4
            valid_flag_list: B x N_of_Resolutions x Anchor_per_location*N_of_locations
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = []
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i])
            multi_level_anchors.append(anchors)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.anchor_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def loss_single(
            self,
            cls_score,
            bbox_pred,
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            num_total_samples,
    ):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_total_samples)

        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples
        )

        return loss_cls, loss_bbox

    def loss(
            self,
            cls_scores,
            bbox_preds,
            gt_bboxes,
            gt_labels,
            img_metas,
            cfg,
            gt_bboxes_ignore=None
    ):
        """
        Args:
            cls_scores (list[tuple]): N(umber of slices for getting anchors) x C x H x W.
            bbox_preds (list[dict]): Image meta info.
            gt_bboxes (list[tuple]): Multi-level feature map sizes.
            gt_labels (list[dict]): Image meta info.
            img_metas (list[dict]): Image meta info.
            cfg (list[dict]): Image meta info.
            gt_bboxes_ignore (list[tuple]): Multi-level feature map sizes.

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas)
        label_channels = 1
        cls_reg_targets = anchor_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg
        )
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)