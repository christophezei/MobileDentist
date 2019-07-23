import os
import mmcv
from PIL import Image
import numpy as np

KEY_MAP = {
    'Periodontal_disease': 1,
    'periodontitis': 1,
    'gingivitis': 1,
    'abscess': 2,
    'caries': 3,
    'wedge_shaped_defect': 4,
    'calculus': 5,
}

# opencv2 coordinate
# 0/0---column (y)--->
#  |
#  |
# row (x)
#  |
#  |
#  v

# PIL coordinate
# 0/0---column/width (x)--->
#  |
#  |
# row/height (y)
#  |
#  |
#  v

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

dir_path = os.path.dirname(os.getcwd())

labelbox_file = dir_path + '/cleaning/export-2019-07-11T20_41_53.065Z.json'
out_file = dir_path + '/cleaning/export-2019-07-11T20_41_53_coco.pickle'
image_dir = dir_path + '/cleaning/711/'


def main():

    setcheck = set()

    annotations = []

    items = mmcv.load(labelbox_file)
    print('generating annotation files for {} images'.format(len(items)))

    for item in items:
        filename = '{}.jpg'.format(item['DataRow ID'])
        print('start for file {}'.format(filename))

        image_file = image_dir + filename
        image = Image.open(image_file)

        width, height = image.size

        bboxes = []
        labels = []
        item_labels = item['Label']

        # classifications
        if item_labels == 'Skip':
            pigment = 0
            soft_deposit = 0
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0,))
        elif bool(item_labels) is False:
            pigment = 0
            soft_deposit = 0
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0,))
        else:
            image_problems = item_labels['image_problems']
            pigment = 1 if 'pigment' in image_problems else 0
            soft_deposit = 1 if 'soft_deposit' in image_problems else 0
            if 'periodontitis' in item_labels.keys():
                for each_finding in item_labels['periodontitis']:
                    points = each_finding['geometry']
                    xmin = int(min([point['x'] for point in points]))
                    ymin = int(min([point['y'] for point in points]))
                    xmax = int(max([point['x'] for point in points]))
                    ymax = int(max([point['y'] for point in points]))
                    box = [xmin, ymin, xmax, ymax]
                    bboxes.append(box)
                    labels.append(int(KEY_MAP['periodontitis']))

            if 'gingivitis' in item_labels.keys():
                for each_finding in item_labels['gingivitis']:
                    points = each_finding['geometry']
                    xmin = int(min([point['x'] for point in points]))
                    ymin = int(min([point['y'] for point in points]))
                    xmax = int(max([point['x'] for point in points]))
                    ymax = int(max([point['y'] for point in points]))
                    box = [xmin, ymin, xmax, ymax]
                    bboxes.append(box)
                    labels.append(int(KEY_MAP['gingivitis']))

            if 'abscess' in item_labels.keys():
                for each_finding in item_labels['abscess']:
                    points = each_finding['geometry']
                    xmin = int(min([point['x'] for point in points]))
                    ymin = int(min([point['y'] for point in points]))
                    xmax = int(max([point['x'] for point in points]))
                    ymax = int(max([point['y'] for point in points]))
                    box = [xmin, ymin, xmax, ymax]
                    bboxes.append(box)
                    labels.append(int(KEY_MAP['abscess']))

            if 'caries' in item_labels.keys():
                for each_finding in item_labels['caries']:
                    points = each_finding['geometry']
                    xmin = int(min([point['x'] for point in points]))
                    ymin = int(min([point['y'] for point in points]))
                    xmax = int(max([point['x'] for point in points]))
                    ymax = int(max([point['y'] for point in points]))
                    box = [xmin, ymin, xmax, ymax]
                    bboxes.append(box)
                    labels.append(int(KEY_MAP['caries']))

            if 'wedge-shaped defect' in item_labels.keys():
                for each_finding in item_labels['wedge-shaped defect']:
                    points = each_finding['geometry']
                    xmin = int(min([point['x'] for point in points]))
                    ymin = int(min([point['y'] for point in points]))
                    xmax = int(max([point['x'] for point in points]))
                    ymax = int(max([point['y'] for point in points]))
                    box = [xmin, ymin, xmax, ymax]
                    bboxes.append(box)
                    labels.append(int(KEY_MAP['wedge_shaped_defect']))

            if 'calculus' in item_labels.keys():
                for each_finding in item_labels['calculus']:
                    points = each_finding['geometry']
                    xmin = int(min([point['x'] for point in points]))
                    ymin = int(min([point['y'] for point in points]))
                    xmax = int(max([point['x'] for point in points]))
                    ymax = int(max([point['y'] for point in points]))
                    box = [xmin, ymin, xmax, ymax]
                    bboxes.append(box)
                    labels.append(int(KEY_MAP['calculus']))

            if not bboxes:
                bboxes = np.zeros((0, 4))
                labels = np.zeros((0,))
            else:
                bboxes = np.array(bboxes, ndmin=2)
                labels = np.array(labels)

        annotation = {
            'filename': filename,
            'width': width,
            'height': height,
            'pigment': int(pigment),
            'soft_deposit': int(soft_deposit),
            'ann': {
                'bboxes': bboxes.astype(np.float32),
                'labels': labels.astype(np.int32),
            }
        }

        annotations.append(annotation)
        print('finish for file {}'.format(filename))

        if filename in setcheck:
            raise RuntimeError('{} already in the set'.format(filename))
        else:
            setcheck.add(filename)

    mmcv.dump(annotations, out_file)


if __name__ == '__main__':
    main()
