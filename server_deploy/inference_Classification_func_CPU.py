import os
import torch

from classifiers.GradCam import GradCAM, to_heatmap
from classifiers.VGG16_Classifier import VGGClassifier
from utils.checkpoint import load_checkpoint
from cores.pre_processing.image_transform_de import ImageTransform, image_transfer_back
from utils.image_utils import to_tensor

dir_path = os.path.dirname(os.getcwd())

# checkpoints
checkpoint_file = dir_path + '/MobileDentist/checkpoints/dental_711_w_fix_SSD_classification/epoch_300.pth'

# classes
CLASSES = (
    'pigment', 'soft_deposit',
)

# device
# device = 'cuda:0'
device = 'cpu'

# input
img_path = dir_path+'/demo/cjtjj8pofbb5b0bqp35txi9dq.jpg'

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


def inference_func_CPU(raw_img):
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

    img_transform = ImageTransform(
        mean=img_transform_cfg['mean'],
        std=img_transform_cfg['std'],
        to_rgb=img_transform_cfg['to_rgb'],
    )

    # inference
    # transform image
    img, img_shape, pad_shape, scale_factor = img_transform(
        img=raw_img,
        scale=img_scale,
        flip=img_transform_cfg['flip'],
        pad_val=img_transform_cfg['pad_values'],
        keep_ratio=img_transform_cfg['resize_keep_ratio'],
    )
    img = to_tensor(img).to(device).unsqueeze(0)

    target_layers = ["backbone.features.30"]
    gcam = GradCAM(model=model)
    probs = gcam.forward(img)
    final_probs = probs.cpu().numpy()

    # heatmap for class 0
    target_class = 0
    ids_ = torch.LongTensor([[target_class]]).to(device)
    gcam.backward(ids=ids_)

    # Grad-CAM
    # regions: [H, W]
    # raw_img: [H, W, 3]
    # ori_regions_heatmap: [H, W, 3]
    for target_layer in target_layers:
        regions_0 = gcam.generate(target_layer=target_layer)
        regions_0 = regions_0[0, 0].cpu().numpy()
        ori_regions_0 = image_transfer_back(
            img=regions_0,
            scale=scale_factor,
            cur_shape=regions_0.shape,
            ori_shape=raw_img.shape[0:2]
        )
        ori_regions_heatmap_0 = to_heatmap(ori_regions_0)

    # heatmap for class 1
    target_class = 1
    ids_ = torch.LongTensor([[target_class]]).to(device)
    gcam.backward(ids=ids_)

    for target_layer in target_layers:
        regions_1 = gcam.generate(target_layer=target_layer)
        regions_1 = regions_1[0, 0].cpu().numpy()
        ori_regions_1 = image_transfer_back(
            img=regions_1,
            scale=scale_factor,
            cur_shape=regions_1.shape,
            ori_shape=raw_img.shape[0:2]
        )
        ori_regions_heatmap_1 = to_heatmap(ori_regions_1)

    return final_probs, ori_regions_0, ori_regions_1, ori_regions_heatmap_0, ori_regions_heatmap_1
