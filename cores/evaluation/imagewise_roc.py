import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import csv

import os

dir_path = os.path.dirname(os.getcwd())


def match(
        results,
        truths,
        num_class,
):
    """
    I for number of images.
    N for number of boxes in results.
    M for number of gtboxes in truths.

    :param results:
[
    [
        [
            [
              x, y, x, y, prob
            ] x number-of-bboxes
        ] x number-of-classes
    ] x number-of-images
]
    :param truths:
[
    {
        'filename': 'a.jpg',
        'width': 1280,
        'height': 720,
        "pigment": int,
        "soft_deposit": int,
        'ann': {
            'bboxes': <np.ndarray> (n, 4 (xmin, ymin, xmax, ymax)),
            'labels': <np.ndarray> (n, ),
        }
    } x number-of-images
]
    :param classes:
    :return:
    """

    maxiou_confidence = []

    for cat in range(num_class):
    # each class
        image_iou_array = np.array([])
        for result, truth in zip(results, truths):
        # each image
            max_prob = 0
            for box in result[cat]:
            # each box
                x_min, y_min, x_max, y_max, prob = box
                max_prob = max(prob, max_prob)

            # N x 1. 1 for max prob of bboxes for each image.
            image_iou_array = np.append(image_iou_array, max_prob)

        image_iou_array = image_iou_array.reshape((-1, 1))
        maxiou_confidence.append(image_iou_array)

    return maxiou_confidence


def plot(maxiou_confidence, truths, num_class, classes_in_results, classes_in_dataset):
    """
    :param maxiou_confidence: np.array
    :param num_groundtruthbox: int
    """
    all_auc =[]
    all_AP =[]
    for i in range(num_class):
        category = classes_in_results[i]
        label_in_dataset = classes_in_dataset.index(category) + 1
        gt = np.array([])
        for truth in truths:
            gt_labels = truth['ann']['labels']
            if label_in_dataset in gt_labels:
                gt = np.append(gt, True)
            else:
                gt = np.append(gt, False)

        auc = metrics.roc_auc_score(
            y_true=gt,
            y_score=maxiou_confidence[i][:, 0],
        )
        fpr, tpr, thresholds = metrics.roc_curve(
            y_true=gt,
            y_score=maxiou_confidence[i][:, 0],
        )

        with open(dir_path+'/visualization/imagewise_roc_{}.csv'.format(i), 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(thresholds)
            writer.writerow(fpr)
            writer.writerow(tpr)

        precision, recall, thresholds = metrics.precision_recall_curve(
            y_true=gt,
            probas_pred=maxiou_confidence[i][:, 0],
        )
        AP = metrics.average_precision_score(
            y_true=gt,
            y_score=maxiou_confidence[i][:, 0],
        )

        plt.figure()
        plt.title('ROC')
        plt.xlabel('False Positive rate')
        plt.ylabel('True Positive rate')
        plt.ylim(0, 1)
        plt.plot(fpr, tpr, label='AUC: ' + str(auc))
        plt.legend()
        plt.savefig(dir_path + '/visualization/imagewise_roc_{}.png'.format(i))

        plt.figure()
        plt.title('Precision-Recall')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.axis([0, 1, 0, 1])
        plt.plot(recall, precision, label='mAP: ' + str(AP))
        plt.savefig(dir_path + '/visualization/imagewise_prc_{}.png'.format(i))
        print('auc for class {} is: {}'.format(i, auc))
        print('AP for class {} is: {}'.format(i, AP))

        all_auc.append(auc)
        all_AP.append(AP)

    print('average auc for all classes is: {}'.format(np.mean(all_auc)))
    print('average AP for all classes is: {}'.format(np.mean(all_AP)))


def plot_imagewise_roc(
        results,
        truths,
        num_class,
        classes_in_results,
        classes_in_dataset,
):
    """
    :param results:
[
    [
        [
            [
              x, y, x, y, prob
            ] x number-of-bboxes
        ] x number-of-classes
    ] x number-of-images
]
    :param truths:
[
    {
        'filename': 'a.jpg',
        'width': 1280,
        'height': 720,
        "pigment": int,
        "soft_deposit": int,
        'ann': {
            'bboxes': <np.ndarray> (n, 4 (xmin, ymin, xmax, ymax)),
            'labels': <np.ndarray> (n, ),
        }
    } x number-of-images
]
    :return:
    """

    assert len(results) == len(truths)
    maxiou_confidence = match(results, truths, num_class)
    plot(maxiou_confidence, truths, num_class, classes_in_results, classes_in_dataset)



