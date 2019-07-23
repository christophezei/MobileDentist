import mmcv
import numpy as np
from cocoapi_rewrite.PythonAPI.pycocotools.coco import COCO
from cocoapi_rewrite.PythonAPI.pycocotools.cocoeval import COCOeval
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval

from cores.evaluation.recall import eval_recalls


def coco_eval(result_file, result_type, coco, iou_thrs, max_dets):
    """

    :param result_file: path to the json file.
    :param result_type:
        'proposal': bbox without considering class.
        'bbox': bbox with considering class.
        'segm':.
        'keypoints':.
    :param coco: dataset type.
    :param iou_thrs: thresholds to be considered as detected.
        default for coco is [0.5:0.05:0.95]

    :param max_dets: max number of detection to be considered as bbox.
        default for coco is [1, 10, 100]
    """
    assert result_type in [
        'proposal', 'bbox', 'segm', 'keypoints'
    ]

    if mmcv.is_str(coco):
        coco = COCO(coco)
    assert isinstance(coco, COCO)

    coco_dets = coco.loadRes(result_file)
    img_ids = coco.getImgIds()

    if result_type == 'segm':
        cocoEval = COCOeval(coco, coco_dets, 'segm')
    elif result_type == 'keypoints':
        cocoEval = COCOeval(coco, coco_dets, 'keypoints')
    else:
        cocoEval = COCOeval(coco, coco_dets, 'bbox')

    cocoEval.params.iouThrs = iou_thrs
    cocoEval.params.maxDets = max_dets

    cocoEval.params.imgIds = img_ids
    if result_type == 'proposal':
        cocoEval.params.useCats = 0

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def fast_eval_recall(
        result_file,
        coco,
        max_dets,
        iou_thrs=np.arange(0.5, 0.96, 0.05),
        print_summary=True
):
    """

    :param result_file: file path to json.
    :param coco: dataset type.
    :param max_dets: number of detections to be considered.
    :param iou_thrs: iou to be considered to be detected.
    :param print_summary (bool).
    :return:
    """

    if mmcv.is_str(coco):
        coco = COCO(coco)
    assert isinstance(coco, COCO)

    if mmcv.is_str(result_file):
        results = mmcv.load(result_file)

    gt_bboxes = []
    img_ids = coco.getImgIds()
    for i in range(len(img_ids)):
        ann_ids = coco.getAnnIds(imgIds=img_ids[i])
        ann_info = coco.loadAnns(ann_ids)
        if len(ann_info) == 0:
            gt_bboxes.append(np.zeros((0, 4)))
            continue
        bboxes = []
        for ann in ann_info:
            if ann.get('ignore', False) or ann['iscrowd']:
                continue
            x1, y1, w, h = ann['bbox']
            bboxes.append([x1, y1, x1 + w - 1, y1 + h - 1])
        bboxes = np.array(bboxes, dtype=np.float32)
        if bboxes.shape[0] == 0:
            bboxes = np.zeros((0, 4))
        gt_bboxes.append(bboxes)

    recalls = eval_recalls(
        gt_bboxes, results, max_dets, iou_thrs, print_summary=print_summary)
    ar = recalls.mean(axis=1)

    for i, num in enumerate(max_dets):
        print('AR@{}\t= {:.4f}'.format(num, ar[i]))

    return ar


def xyxy2xywh(bbox):
    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0] + 1,
        _bbox[3] - _bbox[1] + 1,
    ]


def proposal2json(dataset, results):
    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        bboxes = results[idx]
        for i in range(bboxes.shape[0]):
            data = dict()
            data['image_id'] = img_id
            data['bbox'] = xyxy2xywh(bboxes[i])
            data['score'] = float(bboxes[i][4])
            data['category_id'] = 1
            json_results.append(data)
    return json_results


def det2json(dataset, results):
    """
    convert results of raw type into formated json file.

    jason format:
    [Number_of_bboxes,
     dict(
        image_id:
        bbox: [x, y, w, h]
        score:
        category_id:
     )]

    :param dataset: dataset type.
    :param results: raw type. sorted by images. [Number_of_images, list]
    """
    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        result = results[idx]
        for label in range(len(result)):
            bboxes = result[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = dataset.cat_ids[label]
                json_results.append(data)
    return json_results


def segm2json(dataset, results):
    bbox_json_results = []
    segm_json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        det, seg = results[idx]
        for label in range(len(det)):
            # bbox results
            bboxes = det[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = dataset.cat_ids[label]
                bbox_json_results.append(data)

            # segm results
            # some detectors use different score for det and segm
            if len(seg) == 2:
                segms = seg[0][label]
                mask_score = seg[1][label]
            else:
                segms = seg[label]
                mask_score = [bbox[4] for bbox in bboxes]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['score'] = float(mask_score[i])
                data['category_id'] = dataset.cat_ids[label]
                segms[i]['counts'] = segms[i]['counts'].decode()
                data['segmentation'] = segms[i]
                segm_json_results.append(data)
    return bbox_json_results, segm_json_results


def results2json(dataset, results, out_file):
    """
    convert results of raw type into formatted json file.
    save detection results to out_file.bbox.json.
    return json.

    :param dataset: dataset type.
    :param results: raw type. sorted by images. [Number_of_images, list]
    :param out_file: output file path.
    """
    if isinstance(results[0], list):
        # detection
        json_results = det2json(dataset, results)
        mmcv.dump(json_results, '{}.{}.json'.format(out_file, 'bbox'))
        mmcv.dump(json_results, '{}.{}.json'.format(out_file, 'proposal'))

    elif isinstance(results[0], tuple):
        # segmentation
        json_results = segm2json(dataset, results)
        mmcv.dump(json_results[0], '{}.{}.json'.format(out_file, 'bbox'))
        mmcv.dump(json_results[0], '{}.{}.json'.format(out_file, 'proposal'))
        mmcv.dump(json_results[1], '{}.{}.json'.format(out_file, 'segm'))

    elif isinstance(results[0], np.ndarray):
        # fixed number of results per image.
        # proposal evaluation.
        json_results = proposal2json(dataset, results)
        mmcv.dump(json_results, '{}.{}.json'.format(out_file, 'proposal'))
    else:
        raise TypeError('invalid type of results')
