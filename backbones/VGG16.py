import logging

import torch.nn as nn

from mmcv.cnn.weight_init import constant_init, normal_init, kaiming_init
from utils.checkpoint import load_checkpoint


def conv3x3(in_planes, out_planes, dilation):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        padding=dilation,
        dilation=dilation
    )


def make_vgg_layer(
        in_planes,
        out_planes,
        num_blocks,
        dilation,
        with_bn,
        ceil_mode,
):
    layers = []
    for _ in range(num_blocks):
        layers.append(conv3x3(in_planes, out_planes, dilation))
        if with_bn:
            layers.append(nn.BatchNorm2d(out_planes))
        layers.append(nn.ReLU(inplace=True))
        in_planes = out_planes
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode))

    return layers


class VGG(nn.Module):
    """VGG backbone.

    ~134,000,000 params


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

    Args:
        with_bn (bool): Use BatchNorm or not.
        num_classes (int): number of classes for classification. -1 for non FC layers.
        num_stages (int): VGG stages, normally 5.
        dilations (Sequence[int]): Dilation of each stage. (1, 1, 1, 1, 1) for no dilation.
        out_indices (Sequence[int]): Output from which stages. (0, 1, 2, 3, 4) for every stage.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 for
            not freezing any parameters.
        bn_eval (bool): Whether to set BN layers as eval mode, namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of BN layers.
        ceil_mode (bool): how to resolve downsampling %2 issue.
        with_last_pool (bool): whether to pool the last conv layer.
    """

    arch_settings = [2, 2, 3, 3, 3]
    plane_settings = [64, 128, 256, 512, 512]

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
            with_last_pool
    ):
        super(VGG, self).__init__()

        assert len(dilations) == num_stages
        assert max(out_indices) <= num_stages

        self.num_classes = num_classes
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.in_planes = 3
        self.range_sub_modules = []

        start_idx = 0
        vgg_layers = []
        for i, num_blocks in enumerate(self.arch_settings):
            num_modules = num_blocks * (2 + with_bn) + 1
            end_idx = start_idx + num_modules
            dilation = dilations[i]
            out_planes = self.plane_settings[i]
            vgg_layer = make_vgg_layer(
                in_planes=self.in_planes,
                out_planes=out_planes,
                num_blocks=num_blocks,
                dilation=dilation,
                with_bn=with_bn,
                ceil_mode=ceil_mode
            )
            vgg_layers.extend(vgg_layer)
            self.in_planes = out_planes
            self.range_sub_modules.append([start_idx, end_idx])
            start_idx = end_idx

        if not with_last_pool:
            vgg_layers.pop(-1)
        self.add_module('features', nn.Sequential(*vgg_layers))

        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(model=self, filename=pretrained, map_location=None, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        outs = []
        vgg_layers = getattr(self, self.module_name)
        for i, num_blocks in enumerate(self.arch_settings):
            for j in range(*self.range_sub_modules[i]):
                vgg_layer = vgg_layers[j]
                x = vgg_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if self.num_classes > 0:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(VGG, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        vgg_layers = self.features
        if mode and self.frozen_stages >= 0:
            for i in range(self.frozen_stages):
                for j in range(*self.range_sub_modules[i]):
                    mod = vgg_layers[j]
                    mod.eval()
                    for param in mod.parameters():
                        param.requires_grad = False


