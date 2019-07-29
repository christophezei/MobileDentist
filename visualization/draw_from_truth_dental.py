import os
from utils.image import imshow_det_bboxes
import mmcv
from mmcv.image import imread
from cores.misc import dental_detection_classes
import numpy as np

dir_path = os.path.dirname(os.getcwd())
# image_path = dir_path + '/cleaning/711/cjt1jtcw5oaaj0bqp9t21rh8t.jpg'
image_path = dir_path + '/cleaning/711_converted/'
truth_path = dir_path + '/cleaning/export-2019-07-11T20_41_53_coco.pickle'
save_path = dir_path + '/demo/results/'


def show_one_image(truth, file_path, output_dir):

    found_image_bboxes = []
    found_image_bbox_labels = []

    file_name = file_path.split('/')[-1].split('.')[0]

    for per_truth in truth:
        if per_truth['filename'] == file_name:
            found_image_bboxes = per_truth['ann']['bboxes']
            found_image_bbox_labels = per_truth['ann']['labels'] - 1

    if len(found_image_bbox_labels) == 0:
        found_image_bboxes = np.zeros((0, 4))
        found_image_bbox_labels = np.zeros((0,))

    img = imread(file_path)
    height, width, _ = img.shape

    imshow_det_bboxes(
        img=file_path,
        bboxes=found_image_bboxes,
        # labels should be 0 based.
        labels=found_image_bbox_labels,
        class_names=dental_detection_classes(),
        # no thr. show all.
        score_thr=0,
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
        out_file=output_dir + '{}_result.jpg'.format(file_name)
    )


def main():

    truth = mmcv.load(truth_path)

    # one image, save results to save_path
    if os.path.isfile(image_path):

        print('plotting for image {}'.format(image_path))

        show_one_image(
            truth=truth,
            file_path=image_path,
            output_dir=save_path,
        )

    # a dir of images. save results to save_path/dir_name/
    else:

        print('plotting for directory {}'.format(image_path))

        dir_name = image_path.split('/')[-2]
        output_dir = save_path + dir_name + '/'
        mmcv.mkdir_or_exist(output_dir)

        file_list = []
        for file in os.listdir(image_path):
            if file.endswith(".jpg"):
                file_list.append(os.path.join(image_path, file))

        count = 1
        for file in file_list:
            show_one_image(
                truth=truth,
                file_path=file,
                output_dir=output_dir,
            )
            print(count)
            count = count + 1


if __name__ == '__main__':
    main()
