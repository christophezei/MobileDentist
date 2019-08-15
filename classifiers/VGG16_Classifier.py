import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

from backbones.VGG16 import VGG
from mmcv.cnn import constant_init, kaiming_init, normal_init


class VGGClassifier(nn.Module):
    """
    SSDVGG(
        (features): Sequential(
          (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): ReLU(inplace=True)
          (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (3): ReLU(inplace=True)
          (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
          (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (6): ReLU(inplace=True)
          (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (8): ReLU(inplace=True)
          (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
          (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (11): ReLU(inplace=True)
          (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (13): ReLU(inplace=True)
          (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (15): ReLU(inplace=True)
          (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
          (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (18): ReLU(inplace=True)
          (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (20): ReLU(inplace=True)
          (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (22): ReLU(inplace=True)
          (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
          (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (25): ReLU(inplace=True)
          (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (27): ReLU(inplace=True)
          (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (29): ReLU(inplace=True)
          (30): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
          (31): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6))
          (32): ReLU(inplace=True)
          (33): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
          (34): ReLU(inplace=True)
        )
        (extra): Sequential(
          (0): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
          (1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (2): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
          (3): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (4): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
          (5): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
          (6): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
          (7): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
        )
        (l2_norm): L2Norm()
    """
    def __init__(
            self,
            with_bn,
            num_classes,
            num_stages,
            dilations,
            out_indices,
            frozen_stages,
            bn_eval,
            bn_frozen,
            ceil_mode,
            with_last_pool,
            dimension_before_fc,
            dropout_rate,
            pos_loss_weights,
    ):
        super(VGGClassifier, self).__init__()
        # backbone
        self.pos_loss_weights = pos_loss_weights
        self.backbone = VGG(
            with_bn=with_bn,
            num_classes=-1,
            num_stages=num_stages,
            dilations=dilations,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
            bn_eval=bn_eval,
            bn_frozen=bn_frozen,
            ceil_mode=ceil_mode,
            with_last_pool=with_last_pool
        )
        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * np.prod(dimension_before_fc), 4096),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, num_classes),
        )

        self.CLASSES = None

        # init
        self.init_weights(pretrained=None)

    def init_weights(self, pretrained=None):
        # init for backbone
        self.backbone.init_weights(pretrained=pretrained)
        # init for classifier
        for m in self.classifier.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.01)

    def forward(self, img, img_meta, is_test, **kwargs):
        if is_test:
            return self.forward_test(img, img_meta, **kwargs)
        else:
            return self.forward_train(img, img_meta, **kwargs)

    def forward_train(
            self,
            img,
            img_meta,
            gt_labels,
    ):
        """
        forward for training. return losses.
        """
        x = self.backbone(img)
        x = x.view(x.size(0), -1)
        cls_scores = self.classifier(x)

        losses = self.loss(
            cls_scores=cls_scores,
            gt_labels=gt_labels,
            img_metas=img_meta,
        )

        return losses

    def forward_test(self, img, img_meta, rescale):

        # cls_scores: Number_of_images x num_classes
        x = self.backbone(img)
        x = x.view(x.size(0), -1)
        cls_scores = self.classifier(x)
        sigmoid_cls_scores = F.sigmoid(cls_scores)

        return sigmoid_cls_scores.cpu().numpy()

    def simple_test(self, img):

        # cls_scores: Number_of_images x num_classes
        x = self.backbone(img)
        x = x.view(x.size(0), -1)
        cls_scores = self.classifier(x)

        return cls_scores

    def loss(
            self,
            cls_scores,
            gt_labels,
            img_metas,
    ):
        """
        Args:
            cls_scores: Number_of_images x num_classes
            gt_labels: [Number_of_images, num_classes].
            img_metas: [Number_of_images, list].
        Returns:
            loss_cls: [Number_of_images] of float.
            loss_bbox: [Number_of_images] of float.
        """
        gt_labels = torch.stack(gt_labels)

        loss_cls_all = F.binary_cross_entropy_with_logits(
            input=cls_scores,
            target=gt_labels,
            weight=None,
            reduction='none',
            pos_weight=self.pos_loss_weights,
        )

        return dict(loss_cls=loss_cls_all)
