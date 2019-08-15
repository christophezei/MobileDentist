import os
import mmcv
import numpy as np
import math

dir_path = os.path.dirname(os.getcwd())
DATA_PATH = dir_path + '/cleaning/711_converted/'
ANNOTATION_PAH = dir_path + '/cleaning/export-2019-07-11T20_41_53_coco_2.pickle'
SAVE_PATH = dir_path + '/datasets/dental_711_2/'

TRAIN_RATIO = 0.7

SEED = 3

# MobileDentist annotation file (711train.json)
# [
#     {
#         'filename': 'a',
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


def main():
    mmcv.mkdir_or_exist(SAVE_PATH)

    annotation = mmcv.load(ANNOTATION_PAH)

    # remove empty images
    reduced_annotation = [ann for ann in annotation if ann['ann']['labels'].shape[0] != 0]
    print('we have {} images that are annotated'.format(len(reduced_annotation)))

    # random split into train and test
    num_train = math.ceil(TRAIN_RATIO * len(reduced_annotation))
    np.random.seed(SEED)
    np.random.shuffle(reduced_annotation)
    ann_train = reduced_annotation[0:num_train]
    ann_test = reduced_annotation[num_train:len(reduced_annotation)]
    print('we have {} images for training, {} for testing. '.format(len(ann_train), len(ann_test)))

    # save annotation files
    mmcv.dump(ann_train, SAVE_PATH+'train.pickle')
    mmcv.dump(ann_test, SAVE_PATH + 'test.pickle')


if __name__ == '__main__':
    main()
