import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import csv

import os

dir_path = os.path.dirname(os.getcwd())


def plot_imagewise_classification_roc(
        results,
        truths,
        num_class,
        classes,
):
    """
    :param results:
[
    [
        pigment, soft_deposit,
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

    predict = [[], []]
    for item in results:
        predict[0].append(item[0])
        predict[1].append(item[1])

    gt = [[], []]
    for item in truths:
        gt[0].append(item['pigment'])
        gt[1].append(item['soft_deposit'])

    all_auc = []
    all_AP = []
    for i in range(num_class):
        auc = metrics.roc_auc_score(
            y_true=gt[i],
            y_score=predict[i],
        )
        fpr, tpr, thresholds = metrics.roc_curve(
            y_true=gt[i],
            y_score=predict[i],
        )

        with open(dir_path+'/visualization/imagewise_classification_roc_{}.csv'.format(i), 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(thresholds)
            writer.writerow(fpr)
            writer.writerow(tpr)

        precision, recall, thresholds = metrics.precision_recall_curve(
            y_true=gt[i],
            probas_pred=predict[i],
        )
        AP = metrics.average_precision_score(
            y_true=gt[i],
            y_score=predict[i],
        )

        plt.figure()
        plt.title('ROC')
        plt.xlabel('False Positive rate')
        plt.ylabel('True Positive rate')
        plt.ylim(0, 1)
        plt.plot(fpr, tpr, label='AUC: ' + str(auc))
        plt.legend()
        plt.savefig(dir_path + '/visualization/imagewise_classification_roc_{}.png'.format(i))

        plt.figure()
        plt.title('Precision-Recall')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.axis([0, 1, 0, 1])
        plt.plot(recall, precision, label='mAP: ' + str(AP))
        plt.savefig(dir_path + '/visualization/imagewise_classification_prc_{}.png'.format(i))
        print('auc for class {} is: {}'.format(i, auc))
        print('AP for class {} is: {}'.format(i, AP))

        all_auc.append(auc)
        all_AP.append(AP)

    # class-average
    print('class-average auc is: {}'.format(np.mean(all_auc)))
    print('class-average AP is: {}'.format(np.mean(all_AP)))

    # pure-average
    flat_predict_list = []
    for sublist in predict:
        for item in sublist:
            flat_predict_list.append(item)
    flat_gt_list = []
    for sublist in gt:
        for item in sublist:
            flat_gt_list.append(item)

    auc = metrics.roc_auc_score(
        y_true=flat_gt_list,
        y_score=flat_predict_list,
    )
    fpr, tpr, thresholds = metrics.roc_curve(
        y_true=flat_gt_list,
        y_score=flat_predict_list,
    )

    precision, recall, thresholds = metrics.precision_recall_curve(
        y_true=flat_gt_list,
        probas_pred=flat_predict_list,
    )
    AP = metrics.average_precision_score(
        y_true=flat_gt_list,
        y_score=flat_predict_list,
    )

    plt.figure()
    plt.title('ROC')
    plt.xlabel('False Positive rate')
    plt.ylabel('True Positive rate')
    plt.ylim(0, 1)
    plt.plot(fpr, tpr, label='AUC: ' + str(auc))
    plt.legend()
    plt.savefig(dir_path + '/visualization/imagewise_classification_roc_all.png')

    plt.figure()
    plt.title('Precision-Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.axis([0, 1, 0, 1])
    plt.plot(recall, precision, label='mAP: ' + str(AP))
    plt.savefig(dir_path + '/visualization/imagewise_classification_prc_all.png')
    print('pure average auc is: {}'.format(auc))
    print('pure average AP is: {}'.format(AP))








