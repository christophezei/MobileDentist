import torch

import ops.nms_wrapper as wrapper


def multiclass_nms(
        multi_bboxes,
        multi_scores,
        score_thr,
        nms_cfg,
        max_num=-1,
        score_factors=None
):
    """NMS for multi-class bboxes.
    PERFORM NMS CLASS-INDEPENDENTLY.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class)
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_cfg (List): nms configurations.
            [nms, dets, iou_thr, device_id]
            [soft_nms, dets, iou_thr, method, sigma, min_score]
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.shape[1]
    bboxes, labels = [], []
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
        _scores = multi_scores[cls_inds, i]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)

        if nms_cfg[0] == 'nms':
            cls_dets, _ = wrapper.nms(dets=cls_dets, iou_thr=nms_cfg[1], device_id=nms_cfg[2])
        elif nms_cfg[0] == 'soft_nms':
            cls_dets, _ = wrapper.soft_nms(
                dets=cls_dets, iou_thr=nms_cfg[1], method=nms_cfg[2], sigma=nms_cfg[3], min_score=nms_cfg[4]
            )
        else:
            raise RuntimeError('not supported nms operation type!')

        cls_labels = multi_bboxes.new_full(
            (cls_dets.shape[0], ), i - 1, dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

    return bboxes, labels
