import numpy as np
import mmcv

from cores.misc import dental_detection_classes
from datasets.CustomDataset_de import CustomDataset


class DentalDataset(CustomDataset):
    CLASSES = (
        'Periodontal_disease', 'caries', 'calculus',
    )

    def load_annotations(self, ann_file):

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
        #
        # detection labels:
        # 0: negative
        # 1: Periodontal_disease (includes periodontitis and gingivitis)
        # 2：abscess
        # 3: caries
        # 4: wedge_shaped_defect
        # 5: calculus
        annotations = mmcv.load(ann_file)

        # correct label counts
        # full cat: [
        #   'Periodontal_disease', 'abscess', 'caries', 'wedge_shaped_defect', 'calculus'
        # ]
        cat = dental_detection_classes()
        self.cat2label = {}
        start_count = 1
        for i, cat_name in enumerate(cat):
            if cat_name in self.CLASSES:
                self.cat2label[i+1] = start_count
                start_count = start_count + 1
            else:
                self.cat2label[i + 1] = -1
        print('label-category mapping: {}'.format(self.cat2label))

        return annotations

    def get_ann_info(self, idx):
        ann_info = self.img_infos[idx]['ann']

        gt_bboxes = []
        gt_labels = []

        # filter out useless labels. fix the labels for the training case.
        for bbox, label in zip(ann_info['bboxes'], ann_info['labels']):
            if self.cat2label[label] == -1:
                pass
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[label])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels)

        return ann
