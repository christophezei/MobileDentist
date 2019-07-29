import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def match(
        results,
        truths,
        num_class,
        classes_in_results,
        classes_in_dataset
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

    maxiou_confidence = [[]] * num_class

    for cat in range(num_class):
    # each class
        for result, truth in zip(results, truths):
        # each image

            for box in result[cat]:
            # each box
                iou_array = np.array([])
                x_min, y_min, x_max, y_max, prob = box

                for truth_box, truth_label in zip(truth['ann']['bboxes'], truth['ann']['labels']):

                    cat_in_dataset = classes_in_dataset[truth_label-1]
                    if not (cat_in_dataset in classes_in_results and classes_in_results.index(cat_in_dataset) == cat):
                        continue

                    iou = _cal_IoU(
                        detectedbox=[x_min, y_min, x_max-x_min+1, y_max-y_min+1, prob],
                        groundtruthbox=[truth_box[0], truth_box[1], truth_box[2]-truth_box[0]+1, truth_box[3]-truth_box[1]+1, 1],
                    )
                    iou_array = np.append(iou_array, iou)

                if iou_array.shape[0] == 0:
                    maxiou = 0
                else:
                    maxiou = np.max(iou_array)
                # N x 2. 2 for max_iou and prob.
                maxiou_confidence[cat] = np.append(maxiou_confidence[cat], [maxiou, prob])

        if len(maxiou_confidence[cat]) == 0:
            maxiou_confidence[cat] = np.zeros((0, 2), dtype=np.float32)
        else:
            maxiou_confidence[cat] = maxiou_confidence[cat].reshape(-1, 2)
            maxiou_confidence[cat] = maxiou_confidence[cat][np.argsort(-maxiou_confidence[cat][:, 1])] # prob from largest to smallest

    return maxiou_confidence


def plot(tfs, confidences, num_class):
    """
    :param tf_confidence: np.array
    :param num_groundtruthbox: int
    """

    all_auc =[]
    all_AP =[]
    for i in range(num_class):

        auc = metrics.roc_auc_score(
            y_true=tfs[i],
            y_score=confidences[i],
        )
        fpr, tpr, thresholds = metrics.roc_curve(
            y_true=tfs[i],
            y_score=confidences[i],
        )

        precision, recall, thresholds = metrics.precision_recall_curve(
            y_true=tfs[i],
            probas_pred=confidences[i],
        )
        AP = metrics.average_precision_score(
            y_true=tfs[i],
            y_score=confidences[i],
        )

        import os
        dir_path = os.path.dirname(os.getcwd())

        plt.figure()
        plt.title('ROC')
        plt.xlabel('False Positive rate')
        plt.ylabel('True Positive rate')
        plt.ylim(0, 1)
        plt.plot(fpr, tpr, label='AUC: ' + str(auc))
        plt.legend()
        plt.savefig(dir_path + '/visualization/a_{}.png'.format(i))

        plt.figure()
        plt.title('Precision-Recall')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.axis([0, 1, 0, 1])
        plt.plot(recall, precision, label='mAP: ' + str(AP))
        plt.savefig(dir_path + '/visualization/b_{}.png'.format(i))
        print('auc for class {} is: {}'.format(i, auc))
        print('AP for class {} is: {}'.format(i, AP))

        all_auc.append(auc)
        all_AP.append(AP)

    print('average auc for all classes is: {}'.format(np.mean(all_auc)))
    print('average AP for all classes is: {}'.format(np.mean(all_AP)))


def thres(maxiou_confidence, threshold=0.5):
    """
    :param maxiou_confidence: list of arrays. (class+1) x array[number of detected boxes, 2]
    :param threshold:
    :return tf_confidence: list of arrays.  存放所有检测框对应的tp或fp和置信度
    """

    tfs = []
    confidences = []

    for per_class_maxiou_confidence in maxiou_confidence:

        maxious = per_class_maxiou_confidence[:, 0]
        confidence = per_class_maxiou_confidence[:, 1]
        tf = (maxious >= threshold)

        tfs.append(tf)
        confidences.append(confidence)

    return tfs, confidences


def plot_boxwise_roc(
        results,
        truths,
        threshold,
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
    maxiou_confidence = match(results, truths, num_class, classes_in_results, classes_in_dataset)
    tfs, confidences = thres(maxiou_confidence, threshold)
    plot(tfs, confidences, num_class)


def _cal_IoU(detectedbox, groundtruthbox):
    """
    :param detectedbox: list, [leftx_det, topy_det, width_det, height_det, confidence]
    :param groundtruthbox: list, [leftx_gt, topy_gt, width_gt, height_gt, 1]
    :return iou: 交并比
    """
    leftx_det, topy_det, width_det, height_det, _ = detectedbox
    leftx_gt, topy_gt, width_gt, height_gt, _ = groundtruthbox

    centerx_det = leftx_det + width_det / 2
    centerx_gt = leftx_gt + width_gt / 2
    centery_det = topy_det + height_det / 2
    centery_gt = topy_gt + height_gt / 2

    distancex = abs(centerx_det - centerx_gt) - (width_det + width_gt) / 2
    distancey = abs(centery_det - centery_gt) - (height_det + height_gt) / 2

    if distancex <= 0 and distancey <= 0:
        intersection = distancex * distancey
        union = width_det * height_det + width_gt * height_gt - intersection
        iou = intersection / union
        return iou
    else:
        return 0



