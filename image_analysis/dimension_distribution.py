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

LABEL_PATH = dir_path + '/cleaning/export-2019-07-11T20_41_53_coco.pickle'
OUTPUT_PATH = dir_path + '/image_analysis/711_dimension_distribution.pickle'


def main():

    dimensions = []

    truth = mmcv.load(LABEL_PATH)
    for per_truth in truth:

        width = per_truth['width']
        height = per_truth['height']

        dimensions.append((width, height))

    dump(dimensions, OUTPUT_PATH)


if __name__ == '__main__':
    main()

