from backbones.UNet import UNet


class SSDUNET(UNet):

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
            out_feature_indices=(23, 28, 33, 38, 43)
    ):

        super(SSDUNET, self).__init__(
            with_bn=False,
            num_classes=-1,
            frozen_stages=-1,
            bn_eval=True,
            bn_frozen=False,
            out_feature_indices=out_feature_indices,
        )
