import torch.nn as nn
import logging

from backbones.SSD_VGG16 import SSDVGG
from anchor_heads.SSDHead_COCO import SSDHead
from cores.bbox.transforms import bbox2result


class SSDDetector(nn.Module):
    def __init__(
            self,
            pretrained,
    ):
        super(SSDDetector, self).__init__()

        self.backbone = SSDVGG()
        self.bbox_head = SSDHead(
            # basic
            input_size=300,
            num_classes=81,
            in_channels=(512, 1024, 512, 256, 256, 256),
            # anchor generate
            anchor_ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
            anchor_strides=(8, 16, 32, 64, 100, 300),
            basesize_ratio_range=(0.15, 0.9),
            allowed_border=-1,
            # regression
            target_means=(.0, .0, .0, .0),
            target_stds=(0.1, 0.1, 0.2, 0.2),
            # box assign
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.,
            gt_max_assign_all=False,
            # sampling
            sampling=False,
            # balancing the loss
            neg_pos_ratio=3,
            # loss
            smoothl1_beta=1.,
            # inference nms
            nms_pre=-1,
            score_thr=0.02,
            nms_cfg=['nms', 0.45, None],
            max_per_img=200,
        )
        self.CLASSES = None

        # init
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))
        self.backbone.init_weights(pretrained=pretrained)
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        return x

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)

    def forward(self, img, img_meta, is_test, **kwargs):
        if is_test:
            return self.forward_test(img, img_meta, **kwargs)
        else:
            return self.forward_train(img, img_meta, **kwargs)

    def forward_train(
            self,
            img,
            img_meta,
            gt_bboxes,
            gt_labels,
    ):
        """
        forward for training. return losses.
        """
        x = self.extract_feat(img)
        cls_scores, bbox_preds = self.bbox_head(x)
        losses = self.bbox_head.loss(
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            img_metas=img_meta,
        )

        return losses

    def forward_test(self, img, img_meta, rescale):

        # features: Number_of_levels x Number_of_images x features x H x W
        x = self.extract_feat(img)

        # cls_scores: Number_of_levels x Number_of_images x (num_anchors*num_classes) x H x W
        # bbox_preds: Number_of_levels x Number_of_images x (num_anchors*4) x H x W
        cls_score, bbox_pred = self.bbox_head(x)

        # [Number_of_images, tuple(det_bboxes: k x 4, det_labels: k)].
        bbox_list = self.bbox_head.get_bboxes(
            cls_scores=cls_score,
            bbox_preds=bbox_pred,
            img_metas=img_meta,
            rescale=rescale
        )

        # [Number_of_images, Number_of_classes, (reduced_k, 5)].
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]

        return bbox_results

    def simple_test(self, img, img_meta, rescale=False):

        # features: Number_of_levels x B x features x H x W
        x = self.extract_feat(img)

        # cls_scores: Number_of_levels x B x num_classes x H x W
        # bbox_preds: Number_of_levels x B x num_anchors*4 x H x W
        cls_score, bbox_pred = self.bbox_head(x)

        # [Number_of_images, {det_bboxes: k x 4, det_labels: k}].
        bbox_list = self.bbox_head.get_bboxes(
            cls_scores=cls_score,
            bbox_preds=bbox_pred,
            img_metas=img_meta,
            rescale=rescale
        )

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]
