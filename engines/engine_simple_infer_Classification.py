import os
import os.path as osp
import mmcv
import torch

from classifiers.GradCam import GradCAM
from classifiers.GradCam import save_gradcam, save_gradcam_over_image
from classifiers.VGG16_Classifier import VGGClassifier
from utils.checkpoint import load_checkpoint
from cores.pre_processing.image_transform_de import ImageTransform, image_transfer_back
from utils.image_utils import to_tensor

dir_path = os.path.dirname(os.getcwd())

# checkpoints
checkpoint_file = dir_path + '/work_dirs/dental_711_w_fix_SSD_classification/epoch_300.pth'

# classes
CLASSES = (
    'pigment', 'soft_deposit',
)

# device
# device = 'cuda:0'
device = 'cpu'

# input
img_path = dir_path+'/demo/cjt1jtcw5oaa30bqpn2llahta_cropped.jpg'
fake_img_path = dir_path+'/demo/cjt1jtcw5oaa30bqpn2llahta_result.jpg'

# image
img_scale = (480, 320)
img_transform_cfg = \
    dict(
        mean=[-1, -1, -1],
        std=[1, 1, 1],
        to_rgb=True,
        pad_values=(0, 0, 0),
        flip=False,
        resize_keep_ratio=True,
    )

# output
output_path = dir_path+'/visualization'


def main():
    # define the model and restore checkpoint
    model = VGGClassifier(
        with_bn=False,
        num_classes=len(CLASSES),
        num_stages=5,
        dilations=(1, 1, 1, 1, 1),
        out_indices=(30,),
        frozen_stages=-1,
        bn_eval=True,
        bn_frozen=False,
        ceil_mode=True,
        with_last_pool=True,
        dimension_before_fc=(10, 15),
        dropout_rate=0.5,
        pos_loss_weights=None,
    )

    checkpoint = load_checkpoint(
        model=model,
        filename=checkpoint_file,
        map_location='cpu',
        strict=False,
        logger=None
    )

    # define classes
    model.CLASSES = checkpoint['meta']['CLASSES']

    # put to device and freeze
    model.to(device)
    model.eval()

    # backbone
    # backbone.features
    # backbone.features.0
    # backbone.features.1
    # backbone.features.2
    # backbone.features.3
    # backbone.features.4
    # backbone.features.5
    # backbone.features.6
    # backbone.features.7
    # backbone.features.8
    # backbone.features.9
    # backbone.features.10
    # backbone.features.11
    # backbone.features.12
    # backbone.features.13
    # backbone.features.14
    # backbone.features.15
    # backbone.features.16
    # backbone.features.17
    # backbone.features.18
    # backbone.features.19
    # backbone.features.20
    # backbone.features.21
    # backbone.features.22
    # backbone.features.23
    # backbone.features.24
    # backbone.features.25
    # backbone.features.26
    # backbone.features.27
    # backbone.features.28
    # backbone.features.29
    # backbone.features.30
    # classifier
    # classifier.0
    # classifier.1
    # classifier.2
    # classifier.3
    # classifier.4
    # classifier.5
    # classifier.6
    # for name, module in model.named_modules():
    #     print(name)

    img_transform = ImageTransform(
        mean=img_transform_cfg['mean'],
        std=img_transform_cfg['std'],
        to_rgb=img_transform_cfg['to_rgb'],
    )

    # inference
    # read image
    raw_img = mmcv.imread(img_path)
    ori_shape = raw_img.shape
    print(ori_shape)

    # transform image
    img, img_shape, pad_shape, scale_factor = img_transform(
        img=raw_img,
        scale=img_scale,
        flip=img_transform_cfg['flip'],
        pad_val=img_transform_cfg['pad_values'],
        keep_ratio=img_transform_cfg['resize_keep_ratio'],
    )

    img = to_tensor(img).to(device).unsqueeze(0)
    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=img_transform_cfg['flip']
        )
    ]

    # with torch.no_grad():
    #     result = model.forward_test(img=img, img_meta=img_meta, rescale=False)

    target_layers = ["backbone.features.30"]
    target_class = 0
    gcam = GradCAM(model=model)
    probs = gcam.forward(img)
    ids_ = torch.LongTensor([[target_class]]).to(device)
    gcam.backward(ids=ids_)
    print(probs)

    for target_layer in target_layers:
        print("Generating Grad-CAM @{}".format(target_layer))

        # Grad-CAM
        # regions: [1, 1, H, W]
        # raw_img: [H, W, 3]
        regions = gcam.generate(target_layer=target_layer)
        regions = regions[0, 0].cpu().numpy()

        ori_regions = image_transfer_back(
            img=regions,
            scale=scale_factor,
            cur_shape=regions.shape,
            ori_shape=raw_img.shape[0:2]
        )

        print(ori_regions.shape)

        # save_gradcam(
        #     filename=osp.join(
        #         output_path,
        #         "gradcam-{}.png".format(
        #             target_layer
        #         ),
        #     ),
        #     gcam=regions,
        #     paper_cmap=True,
        # )
        # save_gradcam(
        #     filename=osp.join(
        #         output_path,
        #         "orishape-gradcam-{}.png".format(
        #             target_layer
        #         ),
        #     ),
        #     gcam=ori_regions,
        #     paper_cmap=True,
        # )
        # save_gradcam_over_image(
        #     filename=osp.join(
        #         output_path,
        #         "gradcam_over_img-{}.png".format(
        #             target_layer
        #         ),
        #     ),
        #     gcam=ori_regions,
        #     raw_image=raw_img,
        #     paper_cmap=False,
        # )


if __name__ == '__main__':
    main()



