import mmcv
import numpy as np
from cocoapi_rewrite.PythonAPI.pycocotools.coco import COCO
from cocoapi_rewrite.PythonAPI.pycocotools.cocoeval import COCOeval
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval

from cores.evaluation.recall import eval_recalls


def coco_eval(result_file, result_type, coco, iou_thrs, max_dets, areaRng, areaRngLbl, show_all_labels=False):
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
    cocoEval.params.show_all_labels = show_all_labels

    cocoEval.params.areaRng = areaRng
    cocoEval.params.areaRngLbl = areaRngLbl

    cocoEval.params.imgIds = img_ids
    if result_type == 'proposal':
        cocoEval.params.useCats = 0

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    precision_matrix, recall_matrix, recThrs = cocoEval.eval['precision'], cocoEval.eval['recall'], cocoEval.params.recThrs

    return precision_matrix, recall_matrix, recThrs


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
    :param results: raw type. sorted by images.
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


def det2json_w_groundtruth(truths, results, with_label_only):
    """
    convert results to coco style
    image_id is the index of image, 0 based.
    category id is the label class. 1 based. 1 is the first positive class.

    return jason format:
    [Number_of_bboxes,
     dict(
        image_id:
        bbox: [x, y, w, h]
        score:
        category_id:
     )]

    :param truths:
[
    {
        'filename': 'a.jpg',
        'width': 1280,
        'height': 720,
        "pigment": int,
        "soft_deposit": int,
        'ann': {
            'bboxes': <np.ndarray> (n, 4 (xmin, ymin, xmax, ymax)),
            'labels': <np.ndarray> (n, ),
        }
    } x number-of-images
]
    :param results: raw type.
[
    [
        [
            [
              x, y, x, y, prob
            ] x number-of-bboxes
        ] x number-of-classes
    ] x number-of-images
]
    """
    json_results = []

    if with_label_only is True:
        idx = 0
        for truth, result in zip(truths, results):
            if truth['ann']['labels'].shape[0] == 0:
                continue
            else:
                for label in range(len(result)):
                    bboxes = result[label]
                    for i in range(bboxes.shape[0]):
                        data = dict()
                        data['image_id'] = idx
                        data['bbox'] = xyxy2xywh(bboxes[i])
                        data['score'] = float(bboxes[i][4])
                        data['category_id'] = label + 1
                        json_results.append(data)
                idx = idx + 1
    else:
        for idx, (truth, result) in enumerate(zip(truths, results)):
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = idx
                    data['bbox'] = xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = i+1
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


def results2json_w_groundtruth(truths, results, with_label_only, out_file):
    """
    convert results of raw type into formatted json file.
    save detection results to out_file.bbox.json.
    return json.

    :param truths:
[
    {
        'filename': 'a.jpg',
        'width': 1280,
        'height': 720,
        "pigment": int,
        "soft_deposit": int,
        'ann': {
            'bboxes': <np.ndarray> (n, 4 (xmin, ymin, xmax, ymax)),
            'labels': <np.ndarray> (n, ),
        }
    } x number-of-images
]
    :param results: raw type.
[
    [
        [
            [
              x, y, x, y, prob
            ] x number-of-bboxes
        ] x number-of-classes
    ] x number-of-images
]
    :param out_file: output file path.
    """
    if isinstance(results[0], list):
        # detection

        if len(results) != len(truths):
            raise RuntimeError('different length between results and truths. ')

        json_results = det2json_w_groundtruth(truths, results, with_label_only)

        if out_file is not None:
            mmcv.dump(json_results, '{}.{}.json'.format(out_file, 'bbox'))

        return json_results
    #   mmcv.dump(json_results, '{}.{}.json'.format(out_file, 'proposal'))
    #
    # elif isinstance(results[0], tuple):
    #     # segmentation
    #     json_results = segm2json(dataset, results)
    #     mmcv.dump(json_results[0], '{}.{}.json'.format(out_file, 'bbox'))
    #     mmcv.dump(json_results[0], '{}.{}.json'.format(out_file, 'proposal'))
    #     mmcv.dump(json_results[1], '{}.{}.json'.format(out_file, 'segm'))
    #
    # elif isinstance(results[0], np.ndarray):
    #     # fixed number of results per image.
    #     # proposal evaluation.
    #     json_results = proposal2json(dataset, results)
    #     mmcv.dump(json_results, '{}.{}.json'.format(out_file, 'proposal'))
    else:
        raise TypeError('invalid type of results')


def truths2coco(truths, classes_in_truth, classes_in_dataset, with_label_only, out_file):
    """

    {
    "info": {
        "description": str,
        "url": str,
        "version": str,
        "year": int,
        "contributor": str,
        "date_created": str,
    },
   "licenses":
        [{"url": str, "id": int, "name": str, }, ],
   "images":
        [{"license": int, "file_name": str, "coco_url": str, "height": int, "width": int, "date_captured": str, "flickr_url": str, "id": int}, ],
   "annotations":
        [{"segmentation": [[int, ], ], "area": float, "iscrowd": int, "image_id": int, "bbox": [float (x), float (y), float (w), float (h)], "category_id": int, "id": int}, ],
   "categories":
        [{"supercategory": str, "id": int, "name": str}]
}



    :param truths:
[
    {
        'filename': 'a.jpg',
        'width': 1280,
        'height': 720,
        "pigment": int,
        "soft_deposit": int,
        'ann': {
            'bboxes': <np.ndarray> (n, 4 (xmin, ymin, xmax, ymax)),
            'labels': <np.ndarray> (n, ),
        }
    } x number-of-images
]

    :param classes_in_dataset:
    :param out_file:
    :return:
    """

    START_BOUNDING_BOX_ID = 1

    json_dict = {"images": [], "type": "instances", "annotations": [],
                 "categories": []}

    for ind, per_class in enumerate(classes_in_truth):
        json_dict['categories'].append({"supercategory": per_class, "id": ind+1, "name": per_class})

    image_id = 0
    box_id = START_BOUNDING_BOX_ID
    for truth in truths:

        if with_label_only is True:
            if truth['ann']['labels'].shape[0] == 0:
                continue

        image = dict()
        image['file_name'] = truth['filename']
        image['height'] = truth['height']
        image['width'] = truth['width']
        image['id'] = image_id
        json_dict['images'].append(image)

        for bbox, label in zip(truth['ann']['bboxes'], truth['ann']['labels']):

            # label: 1, 2, 3, 4, 5
            # classes_in_dataset: ['Periodontal_disease', 'abscess', 'caries', 'wedge_shaped_defect', 'calculus']
            # classes_in_truth: ['Periodontal_disease', 'caries', 'calculus']
            cat_from_dataset = classes_in_dataset[label-1]
            if cat_from_dataset not in classes_in_truth:
                continue
            label_from_truth = classes_in_truth.index(cat_from_dataset) + 1

            xmin, ymin, o_width, o_height = xyxy2xywh(bbox)

            ann = {
                'area': o_width * o_height, 'iscrowd': 0, 'image_id':
                image_id, 'bbox': [xmin, ymin, o_width, o_height],
                'category_id': label_from_truth, 'id': box_id, 'ignore': 0,
                'segmentation': []
            }
            json_dict['annotations'].append(ann)

            box_id = box_id + 1

        image_id = image_id + 1

    return json_dict












