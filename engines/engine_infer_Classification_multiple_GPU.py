import argparse
import os
import torch
import mmcv
from functools import partial

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from datasets.DentalClassDataset import DentalClassDataset
from classifiers.VGG16_Classifier import VGGClassifier
from utils.checkpoint import load_checkpoint
from mmcv.parallel.distributed import MMDistributedDataParallel
from mmcv.parallel.collate import collate
from mmcv.runner import get_dist_info
from utils.sampler import NewDistributedSampler
from utils.inference import collect_results

dir_path = os.path.dirname(os.getcwd())
dir_path = dir_path + '/MobileDentist'
print(dir_path)

# checkpoints
checkpoint_file = dir_path + '/work_dirs/dental_711_w_fix_SSD_classification/epoch_300.pth'

# output
# tmp dir for writing some results
tmpdir = dir_path + '/work_dirs/dental_711_w_fix_SSD_classification/tmp/'
# final result dir
out_file = dir_path + '/work_dirs/dental_711_w_fix_SSD_classification/train_data_result'

# input
ann_file = dir_path + '/datasets/dental_711_2/train.pickle'
img_prefix = dir_path + '/cleaning/711_converted/'

# image
img_scale = (480, 320)
flip = False
flip_ratio = 0
img_transform_cfg = \
    dict(
        mean=[-1, -1, -1],
        std=[1, 1, 1],
        to_rgb=True,
        pad_values=(0, 0, 0),
        resize_keep_ratio=True,
    )

# loading
workers_per_gpu = 8
imgs_per_gpu = 8

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
    dataset = DentalClassDataset(
        ann_file=ann_file,
        img_prefix=img_prefix,
        img_scale=img_scale,
        img_norm_cfg=img_transform_cfg,
        multiscale_mode='value',   # select a scale, rather than random from a range.
        flip_ratio=flip_ratio,
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
    model = VGGClassifier(
        with_bn=False,
        num_classes=len(dataset.CLASSES),
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

    checkpoint = load_checkpoint(
        model=model,
        filename=checkpoint_file,
        map_location='cpu',
        strict=False,
        logger=None
    )

    # define classes
    model.CLASSES = checkpoint['meta']['CLASSES']

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


if __name__ == '__main__':
    main()
