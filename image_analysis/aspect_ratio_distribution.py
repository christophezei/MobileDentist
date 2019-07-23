import os
from mmcv import dump
import mmcv

# annotation.pickle coordinate
# 0/0---column/width (x)--->
#  |
#  |
# row/height (y)
#  |
#  |
#  v

# annotation.pickle label
# 0 -> negative
# >= 1 -> positives

# aspect ratio: x / y (width / height)

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

dir_path = os.path.dirname(os.getcwd())

IMG_DIR = dir_path + '/cleaning/711/'
LABEL_PATH = dir_path + '/cleaning/export-2019-07-11T20_41_53_coco.pickle'
OUTPUT_PATH = dir_path + '/image_analysis/711_aspect_ratio_distribution.pickle'


def main():

    aspect_ratios = []

    truth = mmcv.load(LABEL_PATH)
    for per_truth in truth:

        width = per_truth['width']
        height = per_truth['height']

        aspect_ratio = float(width) / float(height)

        if aspect_ratio < 1.0:
            print(per_truth['filename'])

        aspect_ratios.append(aspect_ratio)

    dump(aspect_ratios, OUTPUT_PATH)


if __name__ == '__main__':
    main()

