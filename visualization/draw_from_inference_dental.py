import os
from utils.image import imshow_det_bboxes
import mmcv
from mmcv.image import imread
import numpy as np

CLASSES = ['Periodontal_disease', 'caries', 'calculus']

dir_path = os.path.dirname(os.getcwd())
image_path = dir_path + '/cleaning/711_converted/'
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

result_path = dir_path + '/work_dirs/dental_711_w_pretrained_wt_fix/test_data_result.pickle'
save_path = dir_path + '/work_dirs/dental_711_w_pretrained_wt_fix/test_data_result/'


def show_one_image(result, file_path, output_dir):

    found_image_bboxes = []
    found_image_bbox_labels = []

    file_name = file_path.split('/')[-1]

    for class_label, per_class_result in enumerate(result):
        # class_label: int
        # per_class_result: number-of-bboxes x [x, y, x, y, prob]
        for per_box_per_class_result in per_class_result:
            found_image_bbox_labels.append(class_label)
            found_image_bboxes.append(per_box_per_class_result)

    if len(found_image_bbox_labels) == 0:
        found_image_bboxes = np.zeros((0, 5))
        found_image_bbox_labels = np.zeros((0,))
    else:
        found_image_bboxes = np.asarray(found_image_bboxes).reshape((-1, 5))
        found_image_bbox_labels = np.asarray(found_image_bbox_labels).reshape((-1))

    img = imread(file_path)
    height, width, _ = img.shape

    imshow_det_bboxes(
        img=file_path,
        bboxes=found_image_bboxes,
        # labels should be 0 based.
        labels=found_image_bbox_labels,
        class_names=CLASSES,
        # no thr. show all.
        score_thr=0.3,
        bbox_color='green',
        text_color='green',
        # thickness (int): Thickness of both lines and fonts.
        thickness=np.int((height*width/480/480)**0.5),
        # font_scale (float): Font scales of texts.
        font_scale=np.float((height*width/480/480)**0.5)/2,
        show=False,
        win_name='',
        wait_time=0,
        # save result to a file.
        out_file=output_dir + file_name
    )


def main():

    truths = mmcv.load(truth_path)
    results = mmcv.load(result_path)
    print('plotting for inference {}'.format(result_path))

    mmcv.mkdir_or_exist(save_path)

    count = 1
    for truth, result in zip(truths, results):

        file_path = os.path.join(image_path, truth['filename'])
        show_one_image(
            result=result,
            file_path=file_path,
            output_dir=save_path,
        )
        print(count)
        count = count + 1


if __name__ == '__main__':
    main()
