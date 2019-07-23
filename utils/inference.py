import mmcv
import torch
import numpy as np
import pycocotools.mask as maskUtils
import os.path as osp
import torch.distributed as dist
import shutil

from mmcv.runner import get_dist_info
from utils.image_utils import to_tensor

from visualization.image import imshow_det_bboxes


def _prepare_data(img, img_transform, img_scale, img_resize_keep_ratio, img_flip, device):
    ori_shape = img.shape
    img, img_shape, pad_shape, scale_factor = img_transform(
        img=img,
        scale=img_scale,
        flip=img_flip,
        keep_ratio=img_resize_keep_ratio,
    )

    img = to_tensor(img).to(device).unsqueeze(0)

    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=img_flip
        )
    ]
    return dict(img=[img], img_meta=[img_meta])


def inference_single(model, img, img_transform, scale, flip, resize_keep_ratio, rescale, device):

    img = mmcv.imread(img)

    ori_shape = img.shape
    img, img_shape, pad_shape, scale_factor = img_transform(
        img=img,
        scale=scale,
        flip=flip,
        keep_ratio=resize_keep_ratio,
    )

    img = to_tensor(img).to(device).unsqueeze(0)
    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip
        )
    ]

    with torch.no_grad():
        result = model.forward_test(img=img, img_meta=img_meta, rescale=rescale)

    return result


def show_result(img, result, class_names, score_thr=0.3, out_file=None):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    imshow_det_bboxes(
        img=img.copy(),
        bboxes=bboxes,
        labels=labels,
        class_names=class_names,
        score_thr=score_thr,
        thickness=np.int((img.shape[0]*img.shape[1] / 480 / 480) ** 0.5),
        font_scale=np.float((img.shape[0]*img.shape[1] / 480 / 480) ** 0.5) / 2,
        show=out_file is None,
        out_file=out_file
    )


def collect_results(result_part, dataset_real_size, tmpdir):
    """
    collect results from all gpus and concatenate them into final results.
    Note the results from paddings of dataset are removed.

    :param result_part: result from the current gpu.
    :param dataset_real_size: the real size (unpadded size) of the dataset.
    :param tmpdir: a tmpdir for saving per gpu results. will be removed latter.
    :return: ordered, unpadded results of the whole dataset.
    """
    rank, world_size = get_dist_info()

    # create a tmp dir if it is not specified
    mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    # wait for all gpus to finish.
    dist.barrier()

    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))

        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))

        # the dataloader may pad some samples
        ordered_results = ordered_results[:dataset_real_size]
        # remove tmp dir
        shutil.rmtree(tmpdir)

        return ordered_results