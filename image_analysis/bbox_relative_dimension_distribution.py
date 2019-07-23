import os
import mmcv
from mmcv import dump

dental_detection_classes = \
    ['Periodontal_disease', 'abscess', 'caries', 'wedge_shaped_defect', 'calculus']

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

# aspect ratio: x / y (width / height)

dir_path = os.path.dirname(os.getcwd())

IMG_DIR = dir_path + '/cleaning/711/'
LABEL_PATH = dir_path + '/cleaning/export-2019-07-11T20_41_53_coco.pickle'
OUTPUT_PATH = dir_path + '/image_analysis/711_bbox_relative_dimension_distribution.pickle'


def main():

    bbox_relative_dimension = {}

    for bbox_class in dental_detection_classes:
        bbox_relative_dimension[bbox_class] = []
    bbox_relative_dimension['overall'] = []

    truth = mmcv.load(LABEL_PATH)

    for per_truth in truth:

        width = per_truth['width']
        height = per_truth['height']

        image_bboxes = per_truth['ann']['bboxes']
        image_bbox_labels = per_truth['ann']['labels']

        for box, box_label in zip(image_bboxes, image_bbox_labels):
            relative_width = float(box[2] - box[0]) / float(width)
            relative_height = float(box[3] - box[1]) / float(height)
            bbox_class = dental_detection_classes[int(box_label) - 1]

            bbox_relative_dimension[bbox_class].append((relative_width, relative_height))
            bbox_relative_dimension['overall'].append((relative_width, relative_height))

    dump(bbox_relative_dimension, OUTPUT_PATH)


if __name__ == '__main__':
    main()

