import os
import mmcv

from cores.evaluation.imagewise_roc import plot_imagewise_roc
# from cores.evaluation.boxwise_strict_roc import plot_boxwise_strict_roc
# from cores.evaluation.boxwise_relaxed_roc import plot_boxwise_relaxed_roc
from cores.evaluation.boxwise_relaxed_froc import plot_boxwise_relaxed_froc
# from cores.evaluation.boxwise_strict_froc import plot_boxwise_strict_froc
from cores.misc import dental_detection_classes

CLASSES = ['Periodontal_disease', 'caries', 'calculus']
WITH_LABEL_ONLY = True

dir_path = os.path.dirname(os.getcwd())
# truth_path
# [
#     {
#         'filename': 'a.jpg',
#         'width': 1280,
#         'height': 720,
#         "pigment": int,
#         "soft_deposit": int,
#         'ann': {
#             'bboxes': <np.ndarray> (n, 4 (xmin, ymin, xmax, ymax)),
#             'labels': <np.ndarray> (n, ),
#         }
#     } x number-of-images
# ]
truth_path = dir_path + '/datasets/dental_711_2/test.pickle'
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
result_path = dir_path + '/work_dirs/dental_711_w_pretrained_wt_fix_w_imagenorm_fine_tune_phontrans/test_data_result.pickle'
# img_save_path = dir_path + '/visualization/a.png'


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
    # json_results = results2json_w_groundtruth(truths=truths, results=results, with_label_only=WITH_LABEL_ONLY, out_file=None)
    # print('finish converting model output {} to coco style'.format(result_path))

    # convert truth to coco style
    # image_id is the index of image, 0 based.
    # category id is the label class. 1 based. 1 is the first positive class.
    # {
    #    "images":
    #        [{"height": int, "width": int, "id": int}, ],
    #    "annotations":
    #        [{"area": float, "iscrowd": int, "image_id": int, "bbox": [float (x), float (y), float (w), float (h)], "category_id": int, "id": int}, ],
    #    "categories":
    #        [{"supercategory": str, "id": int, "name": str}]
    # }
    # coco_truths = truths2coco(truths=truths, classes_in_truth=CLASSES, classes_in_dataset=dental_detection_classes(), with_label_only=WITH_LABEL_ONLY, out_file=None)
    # print('finish converting ground truth {} to coco style'.format(truth_path))

    # compare with fast eval
    # T: ioutrhshold
    # R: recall trhsholds
    # K: cats
    # A: object area ranges
    # M: max detections per image
    # precision_matrix: T x R x K x A x M
    # recall_matrix: T x K x A x M
    # recThrs: T
    # precision_matrix, recall_matrix, recThrs = coco_eval(
    #     result_file=json_results,
    #     result_type='bbox',
    #     coco=coco_truths,
    #     iou_thrs=[0.2, 0.3, 0.4, 0.5],
    #     max_dets=[100000],
    #     areaRng=[[0**2, 1e8**2]],
    #     areaRngLbl=['all'],
    #     show_all_labels=True,
    # )

    # plt.plot(recThrs, precision_matrix[0, :, 0, 0, 0], 'ro')
    # plt.savefig(img_save_path)

    print('plot_boxwise_relaxed_froc')
    plot_boxwise_relaxed_froc(
        results=results,
        truths=truths,
        threshold=0.5,
        num_class=3,
        classes_in_results=CLASSES,
        classes_in_dataset=dental_detection_classes(),
        IoRelaxed=True,
    )

    # print('plot_boxwise_relaxed_roc')
    # plot_boxwise_relaxed_roc(
    #     results=results,
    #     truths=truths,
    #     threshold=0.5,
    #     num_class=3,
    #     classes_in_results=CLASSES,
    #     classes_in_dataset=dental_detection_classes(),
    #     IoRelaxed=True
    # )
    #
    # print('plot_boxwise_strict_froc')
    # plot_boxwise_strict_froc(
    #     results=results,
    #     truths=truths,
    #     threshold=0.5,
    #     num_class=3,
    #     classes_in_results=CLASSES,
    #     classes_in_dataset=dental_detection_classes(),
    #     IoRelaxed=True
    # )
    #
    # print('plot_boxwise_strict_roc')
    # plot_boxwise_strict_roc(
    #     results=results,
    #     truths=truths,
    #     threshold=0.5,
    #     num_class=3,
    #     classes_in_results=CLASSES,
    #     classes_in_dataset=dental_detection_classes(),
    #     IoRelaxed=True
    # )

    print('plot_imagewise_roc')
    plot_imagewise_roc(
        results=results,
        truths=truths,
        num_class=3,
        classes_in_results=CLASSES,
        classes_in_dataset=dental_detection_classes(),
    )


if __name__ == '__main__':
    main()
