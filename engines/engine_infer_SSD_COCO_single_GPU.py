import os
import torch
import mmcv
from functools import partial

from torch.utils.data import DataLoader
from datasets.CocoDataset import CocoDataset
from detectors.SSD_VGG16_300_Detector import SSDDetector
from utils.checkpoint import load_checkpoint
from mmcv.parallel.data_parallel import MMDataParallel
from mmcv.parallel.collate import collate

# benchmark
# single gpu 1 workers 1 images / batch 142s  833MB

# single gpu 2 workers 1 images / batch 112s  833MB
# single gpu 2 workers 5 images / batch 93s  991MB
# single gpu 2 workers 10 images / batch  91s 1529MB
# single gpu 2 workers 25 images / batch  88s 2899MB
# single gpu 2 workers 50 images / batch  89s 5253MB
# single gpu 2 workers 100 images / batch  89s 9411MB

# single gpu 1 workers 10 images / batch  200% down
# single gpu 1 workers 5 images / batch  1600%  129s
# single gpu 3 workers 10 images / batch  92s
# single gpu 4 workers 10 images / batch  92s


# eval
from cores.utils.coco_utils import results2json, coco_eval


dir_path = os.path.dirname(os.getcwd())

# checkpoints
checkpoint_file = dir_path + '/checkpoints/ssd300_coco_vgg16_caffe_120e_20181221-84d7110b.pth'

# output
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
imgs_per_gpu = 1
shuffle = False

# set True when input size does not vary a lot
torch.backends.cudnn.benchmark = True


def main():
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

    # define data loader
    batch_size = imgs_per_gpu
    num_workers = workers_per_gpu
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        # when sampler is given, shuffle must be False.
        shuffle=False,
        sampler=None,
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

    # put model onto gpu.
    model = model.cuda()
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    # enumerate all data
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):

        print(data['img_meta'])

        with torch.no_grad():
            result = model(is_test=True, rescale=True, **data)
        results.extend(result)
        print(len(results))

        batch_size = data['img'].size(0)
        for _ in range(batch_size):
            prog_bar.update()

    # write results to file
    # [Number of images, Number of classes, (k, 5)].
    # 5 for t, l, b, r, and prob.
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

        # eval bbox
        #  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.257
        #  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.439
        #  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.262
        #  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.069
        #  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.277
        #  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.426
        #  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.240
        #  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.354
        #  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.376
        #  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.124
        #  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.416
        #  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.585
        # coco_eval(
        #     result_file=out_file+'.bbox.json',
        #     result_type='bbox',
        #     coco=dataset.coco,
        #     iou_thrs=np.arange(0.5, 0.95, 0.05),
        #     max_dets=[1, 10, 100]
        # )
        coco_eval(
            result_file=out_file+'.bbox.json',
            result_type='bbox',
            coco=dataset.coco,
            iou_thrs=[0.5],
            max_dets=[1, 10, 100]
        )


if __name__ == '__main__':
    main()
