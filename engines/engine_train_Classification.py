from __future__ import division

from functools import partial
import torch
import argparse
import logging
import random
import numpy as np
import os
from collections import OrderedDict

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from classifiers.VGG16_Classifier import VGGClassifier
from datasets.DentalClassDataset import DentalClassDataset
from mmcv.runner import get_dist_info
from utils.sampler import DistributedNonGroupSampler
from mmcv.parallel.collate import collate
from mmcv.parallel.distributed import MMDistributedDataParallel
from mmcv.runner import Runner, DistSamplerSeedHook
from utils.distribution import DistOptimizerHook

# benchmark

dir_path = os.path.dirname(os.getcwd())
dir_path = dir_path + '/MobileDentist'
print(dir_path)

# dataset
img_prefix = dir_path + '/cleaning/711_converted/'
ann_file = dir_path + '/datasets/dental_711_2/train.pickle'

# image
img_scale = (480, 320)
flip = True
flip_ratio = 0.5
img_norm_cfg = \
    dict(
        mean=[-1, -1, -1],
        std=[1, 1, 1],
        to_rgb=True,
        pad_values=(0, 0, 0),
        resize_keep_ratio=True,
    )

# Augmentation:
extra_aug = dict(
    photo_metric_distortion=dict(
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18
    ),
    expand=dict(
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 3)
    ),
    random_crop=dict(
        min_ious=(0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.5
    )
)

# log
log_level='INFO'
log_config = dict(
    interval=50,  # 50 iters
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ]
)

# read and save model
work_dir = dir_path + '/work_dirs/dental_711_w_fix_SSD_classification/'
resume_from = None
# load_from = dir_path + '/work_dirs/dental_711_w_pretrained_wt_fix_w_imagenorm_fine_tune_phontrans/epoch_300.pth'
load_from = dir_path + '/work_dirs/dental_711_w_fix_SSD_classification/epoch_100.pth'

# training config
seed = None
do_validation = False
workflow = [('train', 1)]  # [('train', 2), ('val', 1)] means running 2 epochs for training and 1 epoch for validation,
total_epochs = 300

# loading
workers_per_gpu = 8
imgs_per_gpu = 6

# optimizer: SGD
lr = 5e-4
momentum = 0.9
weight_decay = 5e-4
grad_clip = dict(max_norm=35, norm_type=2)

# learning
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[16, 22]
)

# checkpoints saving. interval for how many epoch.
checkpoint_config = dict(interval=25)

# set True when input size does not vary a lot
torch.backends.cudnn.benchmark = True

# train head part only 
train_head_only = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int)
    args = parser.parse_args()
    print('what is the rank of the current program: ')
    print(args.local_rank)

    # initialize dist
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    rank = int(args.local_rank)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    # init logger before other steps
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=log_level
        )
    if args.local_rank != 0:
        logger.setLevel('ERROR')
    logger.info('Starting Distributed training')

    # set random seeds
    if seed is not None:
        logger.info('Set random seed to {}'.format(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # build dataset
    dental_dataset = DentalClassDataset(
        ann_file=ann_file,
        img_prefix=img_prefix,
        img_scale=img_scale,
        img_norm_cfg=img_norm_cfg,
        multiscale_mode='value',   # select a scale, rather than random from a range.
        flip_ratio=flip_ratio,
        with_label=True,
        extra_aug=extra_aug,
        test_mode=False,
    )

    # build model
    model = VGGClassifier(
        with_bn=False,
        num_classes=len(dental_dataset.CLASSES),
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
        pos_loss_weights=torch.tensor((15, 8), dtype=torch.float32, device=torch.device('cuda', rank)),
    )
    model.CLASSES = dental_dataset.CLASSES

    # save class names in
    # checkpoints as meta data
    checkpoint_config['meta'] = dict(
        CLASSES=dental_dataset.CLASSES
    )

    # build sampler for shuffling, padding, and mixing.
    _, world_size = get_dist_info()
    sampler = DistributedNonGroupSampler(
        dataset=dental_dataset,
        samples_per_gpu=imgs_per_gpu,
        num_replicas=world_size,
        rank=args.local_rank,
        shuffle=True,
    )

    # build data loader.
    data_loader = DataLoader(
        dataset=dental_dataset,
        batch_size=imgs_per_gpu,
        # shuffle should be False when sampler is given.
        shuffle=False,
        sampler=sampler,
        batch_sampler=None,
        num_workers=workers_per_gpu,
        collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    )

    # put model on gpus

    # MMDistributedDataParallel(
    #     (module): VGGClassifier(
    #       (backbone): VGG(
    #             (features): Sequential(
    #             (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #             (1): ReLU(inplace=True)
    #             (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #             (3): ReLU(inplace=True)
    #             (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
    #             (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #             (6): ReLU(inplace=True)
    #             (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #             (8): ReLU(inplace=True)
    #             (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
    #             (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #             (11): ReLU(inplace=True)
    #             (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #             (13): ReLU(inplace=True)
    #             (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #             (15): ReLU(inplace=True)
    #             (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
    #             (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #             (18): ReLU(inplace=True)
    #             (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #             (20): ReLU(inplace=True)
    #             (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #             (22): ReLU(inplace=True)
    #             (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
    #             (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #             (25): ReLU(inplace=True)
    #             (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #             (27): ReLU(inplace=True)
    #             (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #             (29): ReLU(inplace=True)
    #             (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
    #           )
    #       )
    #       (classifier): Sequential(
    #           (0): Linear(in_features=12800, out_features=4096, bias=True)
    #           (1): ReLU(inplace=True)
    #           (2): Dropout(p=0.5, inplace=False)
    #           (3): Linear(in_features=4096, out_features=4096, bias=True)
    #           (4): ReLU(inplace=True)
    #           (5): Dropout(p=0.5, inplace=False)
    #           (6): Linear(in_features=4096, out_features=2, bias=True)
    #       )
    #   )
    # )

    model = MMDistributedDataParallel(model.cuda())

    # checkpoint
    # {
    #   meta:
    #   {
    #       mmdet_version:
    #       config: []
    #   }
    #   state_dict:
    #       OrderedDict: [
    #           backbone.features.0.weight: tensor([]),
    #           ...
    #       ]
    # }

    # build optimizer
    if hasattr(model, 'module'):
        pure_model = model.module
    else:
        pure_model = model

    for name, param in pure_model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    if train_head_only:
        train_params = []
        for name, param in pure_model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
            else:
                train_params.append(param)
            
        optimizer = torch.optim.SGD(
            params=train_params,
            lr=lr,
            momentum=momentum,
            dampening=0,
            weight_decay=weight_decay,
            nesterov=False
        )

    else:
        optimizer = torch.optim.SGD(
            params=pure_model.parameters(),
            lr=lr,
            momentum=momentum,
            dampening=0,
            weight_decay=weight_decay,
            nesterov=False
        )

    # build runner: a training helper.
    #   model (:obj:`torch.nn.Module`): The model to be run.
    #   batch_processor (callable): A callable method that process a data
    #       batch. The interface of this method should be
    #       `batch_processor(model, data, train_mode) -> dict`
    #   optimizer (dict or :obj:`torch.optim.Optimizer`).
    #   work_dir (str, optional): The working directory to save checkpoints
    #       and logs.
    #   log_level (int): Logging level.
    #   logger (:obj:`logging.Logger`): Custom logger. If `None`, use the
    #       default logger.
    runner = Runner(
        model=model,
        batch_processor=batch_processor,
        optimizer=optimizer,
        work_dir=work_dir,
        log_level=logging.INFO,
        logger=None,
    )

    # register hooks: optimization after the forward
    optimizer_config = DistOptimizerHook(
        grad_clip=grad_clip,
        coalesce=True,
        bucket_size_mb=-1,
    )
    # register hooks: along with training
    runner.register_training_hooks(
        lr_config=lr_config,
        optimizer_config=optimizer_config,
        checkpoint_config=checkpoint_config,
        log_config=log_config
    )
    # register hooks: set sampler seed before each epoch
    runner.register_hook(DistSamplerSeedHook())

    # resume from: epoch and iter to be continued.
    # load from: start as 0.
    if resume_from is not None:
        runner.resume(resume_from)
    elif load_from is not None:
        runner.load_checkpoint(load_from)

    # data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
    #   and validation.
    # workflow (list[tuple]): A list of (phase, epochs) to specify the
    #   running order and epochs. E.g, [('train', 2), ('val', 1)] means
    #   running 2 epochs for training and 1 epoch for validation,
    #   iteratively.
    # max_epochs (int): Total training epochs.
    runner.run(data_loaders=[data_loader], workflow=workflow, max_epochs=total_epochs)


def batch_processor(model, data, train_mode):
    losses = model(**data, is_test=False)

    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
    log_vars['loss'] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()

    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data)
    )

    return outputs


if __name__ == '__main__':
    main()
