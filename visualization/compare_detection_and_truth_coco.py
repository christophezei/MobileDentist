import os
import mmcv
from cores.utils.coco_utils import results2json_w_groundtruth, truths2coco, coco_eval

from cores.misc import dental_detection_classes

CLASSES = ['Periodontal_disease', 'caries', 'calculus']

dir_path = os.path.dirname(os.getcwd())

truth_path = dir_path + '/datasets/dental_711/train.pickle'
# result_path
# [
#     [
#         [
#             [
#               x, y, x, y, prob
#             ] x number-of-bboxes
#         ] x number-of-classes
#     ] x number-of-images
# ]
result_path = dir_path + '/work_dirs/dental_711_w_pretrained_wt_fix/train_data_result.pickle'

save_path = dir_path + '/work_dirs/dental_711_w_pretrained_wt_fix/train_data_result/'


def main():

    truths = mmcv.load(truth_path)
    results = mmcv.load(result_path)
    print('plotting for inference {}'.format(result_path))

    # convert results to coco style
    # image_id is the index of image, 0 based.
    # category id is the label class. 1 based. 1 is the first positive class.
    # result file that of coco format. save it to result.bbox.json and result.proposal.json.
    # jason format:
    #   [Number_of_bboxes,
    #       dict(
    #           image_id:
    #           bbox: [x, y, w, h]
    #           score:
    #           category_id:
    #       )]
    json_results = results2json_w_groundtruth(truths=truths, results=results, out_file=None)
    print('finish converting model output {} to coco style'.format(result_path))

    # convert truth to coco style
    # image_id is the index of image, 0 based.
    # category id is the label class. 1 based. 1 is the first positive class.
    coco_truths = truths2coco(truths=truths, classes_in_truth=CLASSES, classes_in_dataset=dental_detection_classes(), out_file=None)
    print('finish converting ground truth {} to coco style'.format(truth_path))

    # compare with fast eval
    coco_eval(
        result_file=json_results,
        result_type='bbox',
        coco=coco_truths,
        iou_thrs=[0.2, 0.3, 0.4, 0.5],
        max_dets=[20],
        show_all_labels=False,
    )


if __name__ == '__main__':
    main()
