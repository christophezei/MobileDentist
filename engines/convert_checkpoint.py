import os
import torch
from collections import OrderedDict

DIR_PATH = os.path.dirname(os.getcwd())

INPUT_CHECKPOINT = DIR_PATH + '/checkpoints/ssd300_coco_vgg16_caffe_120e_20181221-84d7110b.pth'
OUTPUT_CHECKPOINT = DIR_PATH + '/checkpoints/ssd300_coco_vgg16_before_head.pth'

ALL_PARAMS =\
    ['backbone.features.0.weight',
     'backbone.features.0.bias',
     'backbone.features.2.weight',
     'backbone.features.2.bias',
     'backbone.features.5.weight',
     'backbone.features.5.bias',
     'backbone.features.7.weight',
     'backbone.features.7.bias',
     'backbone.features.10.weight',
     'backbone.features.10.bias',
     'backbone.features.12.weight',
     'backbone.features.12.bias',
     'backbone.features.14.weight',
     'backbone.features.14.bias',
     'backbone.features.17.weight',
     'backbone.features.17.bias',
     'backbone.features.19.weight',
     'backbone.features.19.bias',
     'backbone.features.21.weight',
     'backbone.features.21.bias',
     'backbone.features.24.weight',
     'backbone.features.24.bias',
     'backbone.features.26.weight',
     'backbone.features.26.bias',
     'backbone.features.28.weight',
     'backbone.features.28.bias',
     'backbone.features.31.weight',
     'backbone.features.31.bias',
     'backbone.features.33.weight',
     'backbone.features.33.bias',
     'backbone.extra.0.weight',
     'backbone.extra.0.bias',
     'backbone.extra.1.weight',
     'backbone.extra.1.bias',
     'backbone.extra.2.weight',
     'backbone.extra.2.bias',
     'backbone.extra.3.weight',
     'backbone.extra.3.bias',
     'backbone.extra.4.weight',
     'backbone.extra.4.bias',
     'backbone.extra.5.weight',
     'backbone.extra.5.bias',
     'backbone.extra.6.weight',
     'backbone.extra.6.bias',
     'backbone.extra.7.weight',
     'backbone.extra.7.bias',
     'backbone.l2_norm.weight',
     'bbox_head.reg_convs.0.weight',
     'bbox_head.reg_convs.0.bias',
     'bbox_head.reg_convs.1.weight',
     'bbox_head.reg_convs.1.bias',
     'bbox_head.reg_convs.2.weight',
     'bbox_head.reg_convs.2.bias',
     'bbox_head.reg_convs.3.weight',
     'bbox_head.reg_convs.3.bias',
     'bbox_head.reg_convs.4.weight',
     'bbox_head.reg_convs.4.bias',
     'bbox_head.reg_convs.5.weight',
     'bbox_head.reg_convs.5.bias',
     'bbox_head.cls_convs.0.weight',
     'bbox_head.cls_convs.0.bias',
     'bbox_head.cls_convs.1.weight',
     'bbox_head.cls_convs.1.bias',
     'bbox_head.cls_convs.2.weight',
     'bbox_head.cls_convs.2.bias',
     'bbox_head.cls_convs.3.weight',
     'bbox_head.cls_convs.3.bias',
     'bbox_head.cls_convs.4.weight',
     'bbox_head.cls_convs.4.bias',
     'bbox_head.cls_convs.5.weight',
     'bbox_head.cls_convs.5.bias'
]

DISCARD_PARAMS = 'bbox_head'

checkpoint = torch.load(INPUT_CHECKPOINT)

if isinstance(checkpoint, OrderedDict):

    state_dict = checkpoint
    print(state_dict.keys())

    new_state_dict = state_dict.copy()

    for key in state_dict.keys():
        if DISCARD_PARAMS in key:
            del new_state_dict[key]
        else:
            pass
    print(new_state_dict.keys())

    torch.save(new_state_dict, OUTPUT_CHECKPOINT)

elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:

    state_dict = checkpoint['state_dict']
    print(state_dict.keys())

    new_state_dict = state_dict.copy()

    for key in state_dict.keys():
        if DISCARD_PARAMS in key:
            del new_state_dict[key]
        else:
            pass
    print(new_state_dict.keys())

    checkpoint['state_dict'] = new_state_dict

    torch.save(checkpoint, OUTPUT_CHECKPOINT)

else:
    raise RuntimeError(
        'No state_dict found in checkpoint file {}'.format(INPUT_CHECKPOINT))