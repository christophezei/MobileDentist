import logging

import torch.nn as nn
import torch.nn.functional as F
import torch

from mmcv.cnn.weight_init import constant_init, kaiming_init
from utils.checkpoint import load_checkpoint


def make_unet_init_layer(
        in_planes,
        out_planes,
        with_bn,
):
    layers = []

    # conv 1
    layers.append(
        nn.Conv2d(
            in_channels=in_planes,
            out_channels=in_planes,
            kernel_size=3,
            padding=1,
            dilation=1,
            bias=True,
        )
    )
    if with_bn:
        layers.append(nn.BatchNorm2d(in_planes))
    layers.append(nn.ReLU(inplace=True))

    # conv 2
    layers.append(
        nn.Conv2d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=3,
            padding=1,
            dilation=1,
            bias=True,
        )
    )
    if with_bn:
        layers.append(nn.BatchNorm2d(out_planes))
    layers.append(nn.ReLU(inplace=True))

    return layers


def make_unet_down_layer(
        in_planes,
        out_planes,
        with_bn,
):
    layers = []

    # down conv
    layers.append(
        nn.Conv2d(
            in_channels=in_planes,
            out_channels=in_planes,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            bias=True,
        )
    )

    # conv 1
    layers.append(
        nn.Conv2d(
            in_channels=in_planes,
            out_channels=in_planes,
            kernel_size=3,
            padding=1,
            dilation=1,
            bias=True,
        )
    )
    if with_bn:
        layers.append(nn.BatchNorm2d(in_planes))
    layers.append(nn.ReLU(inplace=True))

    # conv 2
    layers.append(
        nn.Conv2d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=3,
            padding=1,
            dilation=1,
            bias=True,
        )
    )
    if with_bn:
        layers.append(nn.BatchNorm2d(out_planes))
    layers.append(nn.ReLU(inplace=True))

    return layers


def make_unet_up_layer(
        in_planes,
        bypass_planes,
        out_planes,

        with_bn,
):
    layers = []

    # up conv
    layers.append(
        nn.ConvTranspose2d(
            in_channels=in_planes,
            out_channels=in_planes,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            bias=True,
        )
    )

    # conv 1
    layers.append(
        nn.Conv2d(
            in_channels=(in_planes+bypass_planes),
            out_channels=out_planes,
            kernel_size=3,
            padding=1,
            dilation=1,
            bias=True,
        )
    )
    if with_bn:
        layers.append(nn.BatchNorm2d(out_planes))
    layers.append(nn.ReLU(inplace=True))

    # conv 2
    layers.append(
        nn.Conv2d(
            in_channels=out_planes,
            out_channels=out_planes,
            kernel_size=3,
            padding=1,
            dilation=1,
            bias=True,
        )
    )
    if with_bn:
        layers.append(nn.BatchNorm2d(out_planes))
    layers.append(nn.ReLU(inplace=True))

    return layers


class UNet(nn.Module):
    """UNet backbone.

    ~

    Sequential(
  (0): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU(inplace=True)
  (2): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (3): ReLU(inplace=True)
  (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (5): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (6): ReLU(inplace=True)
  (7): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (8): ReLU(inplace=True)
  (9): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (10): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (11): ReLU(inplace=True)
  (12): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (13): ReLU(inplace=True)
  (14): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (15): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (16): ReLU(inplace=True)
  (17): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (18): ReLU(inplace=True)
  (19): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (21): ReLU(inplace=True)
  (22): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (23): ReLU(inplace=True)
  (24): ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (25): Conv2d(768, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (26): ReLU(inplace=True)
  (27): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (28): ReLU(inplace=True)
  (29): ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (30): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (31): ReLU(inplace=True)
  (32): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (33): ReLU(inplace=True)
  (34): ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (35): Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (36): ReLU(inplace=True)
  (37): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (38): ReLU(inplace=True)
  (39): ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (40): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (41): ReLU(inplace=True)
  (42): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (43): ReLU(inplace=True)
)


    Args:
        with_bn (bool): Use BatchNorm or not.
        num_classes (int): number of classes for classification. -1 for non FC layers.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 for
            not freezing any parameters.
        bn_eval (bool): Whether to set BN layers as eval mode, namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of BN layers.
        ceil_mode (bool): how to resolve downsampling %2 issue.
    """

    arch_settings = ['i', 'd', 'd', 'd', 'd', 'u', 'u', 'u', 'u']
    plane_settings = [32, 64, 128, 256, 512, 256, 128, 64, 32]

    def __init__(
            self,
            with_bn,
            num_classes,
            frozen_stages,
            bn_eval,
            bn_frozen,
            out_feature_indices,
    ):
        super(UNet, self).__init__()

        assert len(self.arch_settings) == len(self.plane_settings)

        self.num_classes = num_classes
        self.frozen_stages = frozen_stages
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.in_planes = 3
        self.range_sub_modules = []
        self.out_feature_indices = out_feature_indices

        start_idx = 0
        unet_layers = []
        for i, block_type in enumerate(self.arch_settings):

            if block_type == 'i':
                num_modules = 2 * (2 + with_bn) # 2 x (conv + bn + activation)
                end_idx = start_idx + num_modules
                out_planes = self.plane_settings[i]
                unet_layer = make_unet_init_layer(
                    in_planes=self.in_planes,
                    out_planes=out_planes,
                    with_bn=with_bn,
                )
                unet_layers.extend(unet_layer)
                self.in_planes = out_planes
                self.range_sub_modules.append([start_idx, end_idx])
                start_idx = end_idx

            elif block_type == 'd':
                num_modules = 1 + 2 * (2 + with_bn)  # downsampling + 2 x (conv + bn + activation)
                end_idx = start_idx + num_modules
                out_planes = self.plane_settings[i]
                unet_layer = make_unet_down_layer(
                    in_planes=self.in_planes,
                    out_planes=out_planes,
                    with_bn=with_bn,
                )
                unet_layers.extend(unet_layer)
                self.in_planes = out_planes
                self.range_sub_modules.append([start_idx, end_idx])
                start_idx = end_idx

            elif block_type == 'u':
                num_modules = 1 + 2 * (2 + with_bn)  # 2 x (conv + bn + activation) + upsampling
                end_idx = start_idx + num_modules
                out_planes = self.plane_settings[i]
                unet_layer = make_unet_up_layer(
                    in_planes=self.in_planes,
                    bypass_planes=self.plane_settings[len(self.plane_settings) - i - 1],
                    out_planes=out_planes,
                    with_bn=with_bn,
                )
                unet_layers.extend(unet_layer)
                self.in_planes = out_planes
                self.range_sub_modules.append([start_idx, end_idx])
                start_idx = end_idx
            else:
                raise RuntimeError('does not support block type of {}'.format(block_type))

        self.add_module('features', nn.Sequential(*unet_layers))

        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_planes,
                    out_channels=num_classes,
                    kernel_size=1,
                    padding=0,
                    dilation=1,
                    bias=False,
                )
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
                elif isinstance(m, nn.ConvTranspose2d):
                    kaiming_init(m)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):

        outputs = []

        short_cuts = []
        unet_layers = self.features
        counter = 0
        for i, block_type in enumerate(self.arch_settings):

            if block_type == 'i':
                for j in range(*self.range_sub_modules[i]):
                    unet_layer = unet_layers[j]
                    x = unet_layer(x)
                    if counter in self.out_feature_indices:
                        outputs.append(x)
                    counter = counter + 1
                short_cuts.append(x)

            elif block_type == 'd':
                for j in range(*self.range_sub_modules[i]):
                    unet_layer = unet_layers[j]
                    x = unet_layer(x)
                    if counter in self.out_feature_indices:
                        outputs.append(x)
                    counter = counter + 1
                short_cuts.append(x)

            elif block_type == 'u':
                mini_counter = 0
                for j in range(*self.range_sub_modules[i]):
                    unet_layer = unet_layers[j]
                    if mini_counter == 1:
                        short_cut = short_cuts[len(self.arch_settings) - i - 1]
                        diff1 = x.size()[3] - short_cut.size()[3]
                        diff2 = x.size()[2] - short_cut.size()[2]
                        short_cut = F.pad(
                            input=short_cut,
                            pad=[diff1 // 2, diff1 - diff1 // 2, diff2 // 2, diff2 - diff2 // 2],
                        )
                        x = torch.cat([x, short_cut], dim=1)
                    x = unet_layer(x)
                    if counter in self.out_feature_indices:
                        outputs.append(x)
                    counter = counter + 1
                    mini_counter = mini_counter + 1

        if self.num_classes > 0:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            outputs.append(x)

        if len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)

    def train(self, mode=True):
        super(UNet, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        unet_layers = self.features
        if mode and self.frozen_stages >= 0:
            for i in range(self.frozen_stages):
                for j in range(*self.range_sub_modules[i]):
                    mod = unet_layers[j]
                    mod.eval()
                    for param in mod.parameters():
                        param.requires_grad = False


