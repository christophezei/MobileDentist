"""
Multi-Process Single-GPU with nn.parallel.DistributedDataParallel

This is the highly recommended way to use DistributedDataParallel, with multiple processes,
each of which operates on a single GPU. This is currently the fastest approach to do data parallel
training using PyTorch and applies to both single-node(multi-GPU) and multi-node data parallel training.

*** It is proven to be significantly faster than torch.nn.DataParallel for single-node multi-GPU data parallel training. ***

Here is how to use it: on each host with N GPUs, you should spawn up N processes,
while ensuring that each process individually works on a single GPU from 0 to N-1.

nccl backend is currently the fastest and highly recommended backend to be used with Multi-Process
Single-GPU distributed training and this applies to both single-node and multi-node distributed training

Parameters are never broadcast between processes.
The module performs an all-reduce step on gradients and assumes that they will be modified by the optimizer in all processes in the same way.
Buffers (e.g. BatchNorm stats) are broadcast from the module in process of rank 0,
to all other replicas in the system in every iteration.


Compare with nn.DataParallel

In the single-machine synchronous case, torch.distributed or the torch.nn.parallel.DistributedDataParallel() wrapper may
still have advantages over other approaches to data-parallelism, including torch.nn.DataParallel():

1. Each process maintains its own optimizer and performs a complete optimization step with each iteration.
While this may appear redundant, since the gradients have already been gathered together and averaged across processes
and are thus the same for every process, this means that no parameter broadcast step is needed,
reducing time spent transferring tensors between nodes.

2. Each process contains an independent Python interpreter,
eliminating the extra interpreter overhead and “GIL-thrashing” that comes from driving several execution threads,
model replicas, or GPUs from a single Python process. This is especially important for models that make heavy use of
the Python runtime, including models with recurrent layers or many small components.

"""

import argparse
import os
import torch
import mmcv
from functools import partial

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from datasets.CocoDataset import CocoDataset
from detectors.SSD_VGG16_300_Detector import SSDDetector
from utils.checkpoint import load_checkpoint
from mmcv.parallel.distributed import MMDistributedDataParallel
from mmcv.parallel.collate import collate
from mmcv.runner import get_dist_info
from utils.sampler import NewDistributedSampler
from utils.inference import collect_results

# benchmark
# single gpu 2 workers 5 images / batch 93s  991MB
# two gpu 2 workers 5 images / batch 50s  991MB
# three gpu 2 workers 5 images / batch 35s  991MB

# single gpu 2 workers 25 images / batch  88s 2899MB
# two gpu 2 workers 25 images / batch  48s 2899MB
# three gpu 2 workers 25 images / batch  35s 2899MB

# single gpu 1 workers 10 images / batch  1*200% down
# three gpu 1 workers 10 images / batch  3*1031% 66s

# single gpu 1 workers 5 images / batch  1*1600%  129s
# three gpu 1 workers 5 images / batch  3*1100%  97s


# eval
from cores.utils.coco_utils import results2json, coco_eval

dir_path = os.path.dirname(os.getcwd())
dir_path = dir_path + '/MobileDentist'
print(dir_path)

# checkpoints
checkpoint_file = dir_path + '/checkpoints/ssd300_coco_vgg16_caffe_120e_20181221-84d7110b.pth'

# output
# tmp dir for writing some results
tmpdir = dir_path + '/demo/tmp/'
# final result dir
out_file = dir_path + '/demo/result'

# input
ann_file = dir_path + '/datasets/coco/annotations/instances_val2017.json'
img_prefix = dir_path + '/datasets/coco/val2017/'

# image
img_scale = (300, 300)
flip = False
flip_ratio = 0
img_transform_cfg = \
    dict(
        mean=[123.675, 116.28, 103.53],
        std=[1, 1, 1],
        to_rgb=True,
        size_divisor=None,
        resize_keep_ratio=False
    )

# loading
workers_per_gpu = 1
imgs_per_gpu = 25

# set True when input size does not vary a lot
torch.backends.cudnn.benchmark = True


def main():

    # get local rank from distributed launcher
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    print('what is the rank of the current program: ')
    print(args.local_rank)

    # initialize dist
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    rank = int(args.local_rank)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    # define dataset
    dataset = CocoDataset(
        ann_file=ann_file,
        img_prefix=img_prefix,
        img_scale=img_scale,
        img_norm_cfg=img_transform_cfg,
        multiscale_mode='value',
        flip_ratio=flip_ratio,
        with_ignore=False,
        with_label=False,
        extra_aug=None,
        test_mode=True,
    )

    # sampler for make number of samples % number of gpu == 0
    rank, world_size = get_dist_info()
    sampler = NewDistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        images_per_gpu=imgs_per_gpu,
        rank=rank,
        shuffle=False
    )

    # data loader. Note this is the code for one (each) gpu.
    batch_size = imgs_per_gpu
    num_workers = workers_per_gpu
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        # when sampler is given, shuffle must be False.
        shuffle=False,
        sampler=sampler,
        batch_sampler=None,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    )

    # define the model and restore checkpoint
    model = SSDDetector(
        pretrained=None
    )
    checkpoint = load_checkpoint(
        model=model,
        filename=checkpoint_file,
        map_location='cpu',
        strict=False,
        logger=None
    )

    # define classes
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    # parallelize model
    model = model.cuda()
    model = MMDistributedDataParallel(
        module=model,
        dim=0,
        broadcast_buffers=True,
        bucket_cap_mb=25
    )
    model.eval()

    # results and progress bar
    results = []
    dataset = data_loader.dataset

    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    # enumerate all data
    for i, data in enumerate(data_loader):

        print(data['img_meta'])

        with torch.no_grad():
            result = model(is_test=True, rescale=True, **data)
        results.extend(result)

        # update program bar only if it is rank 0.
        if rank == 0:
            batch_size = data['img'].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all gpus
    results = collect_results(
        result_part=results,
        dataset_real_size=len(dataset),
        tmpdir=tmpdir
    )

    # write results to file
    # [Number of images, Number of classes, (k, 5)].
    # 5 for t, l, b, r, and prob.
    if rank == 0:
        print('\nwriting results to {}'.format(out_file))
        mmcv.dump(results, out_file+'.pickle')

        if not isinstance(results[0], dict):
            # result file that of coco format. save it to result.bbox.json and result.proposal.json.
            # jason format:
            #   [Number_of_bboxes,
            #       dict(
            #           image_id:
            #           bbox: [x, y, w, h]
            #           score:
            #           category_id:
            #       )]
            results2json(dataset, results, out_file)

            #  Average Precision  (AP) @[ IoU=0.50      | maxDets=  1 ] = 0.310
            #  Average Precision  (AP) @[ IoU=0.50      | maxDets= 10 ] = 0.431
            #  Average Precision  (AP) @[ IoU=0.50      | maxDets=100 ] = 0.443
            #  Average Recall     (AR) @[ IoU=0.50      | maxDets=  1 ] = 0.368
            #  Average Recall     (AR) @[ IoU=0.50      | maxDets= 10 ] = 0.582
            #  Average Recall     (AR) @[ IoU=0.50      | maxDets=100 ] = 0.637
            # coco_eval(
            #     result_file=out_file+'.bbox.json',
            #     result_type='bbox',
            #     coco=dataset.coco,
            #     iou_thrs=[0.5],
            #     max_dets=[1, 10, 100]
            # )

            #  Average Precision  (AP) @[ IoU=0.50      | maxDets=  1 ] = 0.310
            #  Average Precision  (AP) @[ IoU=0.50      | maxDets= 10 ] = 0.431
            #  Average Precision  (AP) @[ IoU=0.50      | maxDets=100 ] = 0.443
            #  Average Precision  (AP) @[ IoU=0.55      | maxDets=  1 ] = 0.298
            #  Average Precision  (AP) @[ IoU=0.55      | maxDets= 10 ] = 0.411
            #  Average Precision  (AP) @[ IoU=0.55      | maxDets=100 ] = 0.421
            #  Average Precision  (AP) @[ IoU=0.60      | maxDets=  1 ] = 0.281
            #  Average Precision  (AP) @[ IoU=0.60      | maxDets= 10 ] = 0.382
            #  Average Precision  (AP) @[ IoU=0.60      | maxDets=100 ] = 0.390
            #  Average Precision  (AP) @[ IoU=0.65      | maxDets=  1 ] = 0.263
            #  Average Precision  (AP) @[ IoU=0.65      | maxDets= 10 ] = 0.350
            #  Average Precision  (AP) @[ IoU=0.65      | maxDets=100 ] = 0.355
            #  Average Precision  (AP) @[ IoU=0.70      | maxDets=  1 ] = 0.238
            #  Average Precision  (AP) @[ IoU=0.70      | maxDets= 10 ] = 0.308
            #  Average Precision  (AP) @[ IoU=0.70      | maxDets=100 ] = 0.312
            #  Average Precision  (AP) @[ IoU=0.75      | maxDets=  1 ] = 0.206
            #  Average Precision  (AP) @[ IoU=0.75      | maxDets= 10 ] = 0.260
            #  Average Precision  (AP) @[ IoU=0.75      | maxDets=100 ] = 0.262
            #  Average Precision  (AP) @[ IoU=0.80      | maxDets=  1 ] = 0.165
            #  Average Precision  (AP) @[ IoU=0.80      | maxDets= 10 ] = 0.201
            #  Average Precision  (AP) @[ IoU=0.80      | maxDets=100 ] = 0.202
            #  Average Precision  (AP) @[ IoU=0.85      | maxDets=  1 ] = 0.111
            #  Average Precision  (AP) @[ IoU=0.85      | maxDets= 10 ] = 0.130
            #  Average Precision  (AP) @[ IoU=0.85      | maxDets=100 ] = 0.130
            #  Average Precision  (AP) @[ IoU=0.90      | maxDets=  1 ] = 0.048
            #  Average Precision  (AP) @[ IoU=0.90      | maxDets= 10 ] = 0.053
            #  Average Precision  (AP) @[ IoU=0.90      | maxDets=100 ] = 0.053
            #  Average Precision  (AP) @[ IoU=0.95      | maxDets=  1 ] = 0.005
            #  Average Precision  (AP) @[ IoU=0.95      | maxDets= 10 ] = 0.005
            #  Average Precision  (AP) @[ IoU=0.95      | maxDets=100 ] = 0.005
            #  Average Recall     (AR) @[ IoU=0.50      | maxDets=  1 ] = 0.368
            #  Average Recall     (AR) @[ IoU=0.50      | maxDets= 10 ] = 0.582
            #  Average Recall     (AR) @[ IoU=0.50      | maxDets=100 ] = 0.637
            #  Average Recall     (AR) @[ IoU=0.55      | maxDets=  1 ] = 0.354
            #  Average Recall     (AR) @[ IoU=0.55      | maxDets= 10 ] = 0.557
            #  Average Recall     (AR) @[ IoU=0.55      | maxDets=100 ] = 0.606
            #  Average Recall     (AR) @[ IoU=0.60      | maxDets=  1 ] = 0.338
            #  Average Recall     (AR) @[ IoU=0.60      | maxDets= 10 ] = 0.523
            #  Average Recall     (AR) @[ IoU=0.60      | maxDets=100 ] = 0.565
            #  Average Recall     (AR) @[ IoU=0.65      | maxDets=  1 ] = 0.319
            #  Average Recall     (AR) @[ IoU=0.65      | maxDets= 10 ] = 0.478
            #  Average Recall     (AR) @[ IoU=0.65      | maxDets=100 ] = 0.509
            #  Average Recall     (AR) @[ IoU=0.70      | maxDets=  1 ] = 0.291
            #  Average Recall     (AR) @[ IoU=0.70      | maxDets= 10 ] = 0.421
            #  Average Recall     (AR) @[ IoU=0.70      | maxDets=100 ] = 0.443
            #  Average Recall     (AR) @[ IoU=0.75      | maxDets=  1 ] = 0.258
            #  Average Recall     (AR) @[ IoU=0.75      | maxDets= 10 ] = 0.360
            #  Average Recall     (AR) @[ IoU=0.75      | maxDets=100 ] = 0.373
            #  Average Recall     (AR) @[ IoU=0.80      | maxDets=  1 ] = 0.215
            #  Average Recall     (AR) @[ IoU=0.80      | maxDets= 10 ] = 0.287
            #  Average Recall     (AR) @[ IoU=0.80      | maxDets=100 ] = 0.295
            #  Average Recall     (AR) @[ IoU=0.85      | maxDets=  1 ] = 0.158
            #  Average Recall     (AR) @[ IoU=0.85      | maxDets= 10 ] = 0.203
            #  Average Recall     (AR) @[ IoU=0.85      | maxDets=100 ] = 0.207
            #  Average Recall     (AR) @[ IoU=0.90      | maxDets=  1 ] = 0.084
            #  Average Recall     (AR) @[ IoU=0.90      | maxDets= 10 ] = 0.101
            #  Average Recall     (AR) @[ IoU=0.90      | maxDets=100 ] = 0.103
            #  Average Recall     (AR) @[ IoU=0.95      | maxDets=  1 ] = 0.014
            #  Average Recall     (AR) @[ IoU=0.95      | maxDets= 10 ] = 0.017
            #  Average Recall     (AR) @[ IoU=0.95      | maxDets=100 ] = 0.017
            coco_eval(
                result_file=out_file + '.bbox.json',
                result_type='bbox',
                coco=ann_file,
                iou_thrs=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                max_dets=[1, 10, 100]
            )


if __name__ == '__main__':
    main()
