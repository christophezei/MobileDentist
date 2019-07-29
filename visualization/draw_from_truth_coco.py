import os
from utils.image import imshow_det_bboxes
import mmcv
from cocoapi_rewrite.PythonAPI.pycocotools.coco import COCO
import numpy as np
from cores.misc import coco_classes

dir_path = os.path.dirname(os.getcwd())
image_path = dir_path + '/datasets/coco/val2017/000000288062.jpg'
truth_path = dir_path + '/datasets/coco/annotations/instances_val2017.json'
save_path = dir_path + '/demo/results/'


def main():

    file_name = image_path.split('/')[-1].split('.')[0]

    # one image, save results to save_path
    if os.path.isfile(image_path):
        truth = mmcv.load(truth_path)
        truth_images = truth['images']

        found_image_id = None
        found_image_bboxes = []
        found_image_bbox_labels = []

        for truth_image in truth_images:
            if truth_image['file_name'] == '{}.jpg'.format(file_name):
                found_image_id = truth_image['id']

        COCO_api = COCO(annotation_file=truth_path)
        found_ann_ids = COCO_api.getAnnIds(imgIds=[found_image_id])
        found_anns = COCO_api.loadAnns(found_ann_ids)

        for found_ann in found_anns:
            found_image_bboxes.append(
                (
                    # annotation file is [topleft_x, topleft_y, w, h]
                    # plot function is [topleft_x, topleft_y, botright_x, botright_y]
                    found_ann['bbox'][0],
                    found_ann['bbox'][1],
                    found_ann['bbox'][0]+found_ann['bbox'][2],
                    found_ann['bbox'][1]+found_ann['bbox'][3],
                )
            )
            found_image_bbox_labels.append(found_ann['category_id'])

        # coco annotation catids are 1 based, 90 in total.
        cat_ids = COCO_api.getCatIds()
        # our label are 1 based, 80 in total.
        cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(cat_ids)
        }

        found_image_bbox_converted_labels = []
        for found_image_bbox_label in found_image_bbox_labels:
            found_image_bbox_converted_labels.append(cat2label[found_image_bbox_label])

        if not found_image_bboxes:
            found_image_bboxes = np.zeros((0, 4))
            found_image_bbox_converted_labels = np.zeros((0,)) - 1
        else:
            found_image_bboxes = np.array(found_image_bboxes, ndmin=2)
            found_image_bbox_converted_labels = np.array(found_image_bbox_converted_labels) - 1

        imshow_det_bboxes(
            img=image_path,
            bboxes=found_image_bboxes,
            # labels should be 0 based.
            labels=found_image_bbox_converted_labels,
            class_names=coco_classes(),
            # no thr. show all.
            score_thr=0,
            bbox_color='green',
            text_color='green',
            thickness=1,
            font_scale=0.5,
            show=False,
            win_name='',
            wait_time=0,
            # save result to a file.
            out_file=save_path+'{}_result.jpg'.format(file_name)
        )

    # a dir of images. save results to save_path/dir_name/
    else:
        pass


if __name__ == '__main__':
    main()
