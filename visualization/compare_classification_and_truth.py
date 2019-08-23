import os
import mmcv

from cores.evaluation.imagewise_classification_roc import plot_imagewise_classification_roc

CLASSES = ['pigment', 'soft_deposit']
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
#       prob_pigment, prob_soft_deposit
#     ] x number-of-images
# ]
result_path = dir_path + '/work_dirs/dental_711_w_fix_SSD_classification/test_data_result.pickle'
# img_save_path = dir_path + '/visualization/a.png'


def main():

    truths = mmcv.load(truth_path)
    results = mmcv.load(result_path)
    print('plotting for inference {}'.format(result_path))

    print('plot_imagewise_classification_roc')
    plot_imagewise_classification_roc(
        results=results,
        truths=truths,
        num_class=2,
        classes=CLASSES,
    )


if __name__ == '__main__':
    main()
