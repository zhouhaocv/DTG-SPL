import numpy as np
from terminaltables import AsciiTable
from core.config import config, update_config
def iou(pred, gt): # require pred and gt is numpy
    assert isinstance(pred, list) and isinstance(gt,list)
    pred_is_list = isinstance(pred[0],list)
    gt_is_list = isinstance(gt[0],list)
    if not pred_is_list: pred = [pred]
    if not gt_is_list: gt = [gt]
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:,0,None], gt[None,:,0])
    inter_right = np.minimum(pred[:,1,None], gt[None,:,1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:,0,None], gt[None,:,0])
    union_right = np.maximum(pred[:,1,None], gt[None,:,1])
    union = np.maximum(0.0, union_right - union_left)
    overlap = 1.0 * inter / union
    if not gt_is_list:
        overlap = overlap[:,0]
    if not pred_is_list:
        overlap = overlap[0]
    return overlap

def nms(dets, thresh=0.4, top_k=-1):
    """Pure Python NMS baseline."""
    if len(dets) == 0: return np.array([])
    order = np.arange(0,len(dets),1)
    dets = np.array(dets)
    x1 = dets[:, 0]
    x2 = dets[:, 1]
    lengths = x2 - x1
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if len(keep) == top_k:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1)
        ovr = inter / (lengths[i] + lengths[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep]

def soft_nms(dets,thresh = 0.4,top_k=-1,method = 'hard', sigma = 0.5):
    """Pure Python Soft_NMS baseline."""
    if len(dets) == 0: return np.array([])
    if method not in ['hard','linear','gaussian']:
        raise ValueError("Unsupported NMS_type: {0}".format(method))
    order = np.arange(0,len(dets),1)
    dets = np.array(dets)
    x1 = dets[:, 0]
    x2 = dets[:, 1]
    scores = dets[:, 2]
    lengths = x2 - x1
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if len(keep) == top_k:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1)
        ovr = inter / (lengths[i] + lengths[order[1:]] - inter)

        weight = np.ones(len(scores)-1)
        if method == 'linear': 
            inds = np.where(ovr > thresh)[0]
            weight[inds] = weight[inds] - ovr[inds]
        elif method == 'gaussian': 
            weight = np.exp(-(ovr * ovr) / sigma)
        else:  
            inds = np.where(ovr > thresh)[0]
            weight[inds] = 0

        # 权重重新调整
        scores = weight * scores[1:] 
        sorted_index = np.argsort(-scores)
        order = order[sorted_index + 1]
        scores = scores[sorted_index]

    return dets[keep]

import matplotlib.pyplot as plt
def evals(segments, data):
    tious = [float(i) for i in config.TEST.TIOU.split(',')] if isinstance(config.TEST.TIOU,str) else [config.TEST.TIOU]
    recalls = [int(i) for i in config.TEST.RECALL.split(',')] if isinstance(config.TEST.RECALL,str) else [config.TEST.RECALL]

    eval_result = [[[] for _ in recalls] for _ in tious]
    max_recall = max(recalls)
    average_iou = []
    for seg, dat in zip(segments, data):
        if len(seg) == 0: 
            overlap=np.array([[0],[0],[0],[0],[0]])
        else:
            # seg = seg2[:,:2].tolist()
            overlap = iou(seg, [dat['times']])
        average_iou.append(np.mean(np.sort(overlap[0])[-3:]))

        for i,t in enumerate(tious):
            for j,r in enumerate(recalls):
                eval_result[i][j].append((overlap >= t)[:r].any())
    eval_result = np.array(eval_result).mean(axis=-1)
    miou = np.mean(average_iou)

    return eval_result, miou

def evals_new(segments, data):
    tious = [float(i) for i in config.TEST.TIOU.split(',')] if isinstance(config.TEST.TIOU,str) else [config.TEST.TIOU]
    recalls = [int(i) for i in config.TEST.RECALL.split(',')] if isinstance(config.TEST.RECALL,str) else [config.TEST.RECALL]
    max_recall = max(recalls)
    beta = config.TEST.EVAL_BETA

    eval_result_NN_Rep = [[[] for _ in tious] for _ in range(2)]
    eval_result_multilabel = [[[] for _ in tious] for _ in range(2)]

    for seg, dat in zip(segments, data):
        overlap_gt = iou(dat['multi_times'], dat['multi_times'])
        max_gt = len(overlap_gt)
        overlap_gt = (overlap_gt.sum(1)-1)/(max_gt-1)
        gt_sorted_index = np.argsort(overlap_gt)[::-1]
        overlap_gt = overlap_gt[gt_sorted_index]

        if len(seg) == 0: 
            overlap_pred_gt=np.zeros((5,max_gt))
        else:
            # seg = seg2[:,:2].tolist()
            overlap_pred_gt = iou(seg, dat['multi_times'])

        overlap_pred_gt = overlap_pred_gt[:,gt_sorted_index]

        #NN REP metrics from uncovering hidden challenges in query-based video moment retrieval
        for j,t in enumerate(tious):
            eval_result_NN_Rep[0][j].append((overlap_pred_gt[0] >= t).any())
            eval_result_NN_Rep[1][j].append((overlap_pred_gt[0][0] >= t).any())

        #R @ (N,G), IoU=alpha
        for j,t in enumerate(tious):
            for k in range(max_gt):
                eval_result_multilabel[0][j].append((overlap_pred_gt[:,k] >= t).any())

        #R_beta @ (N,G), IoU=alpha
        keep = overlap_gt > beta
        num_multi = keep.sum()
        if num_multi <1:
            num_multi = 1
        for j,t in enumerate(tious):
            for i in range(num_multi):
                eval_result_multilabel[1][j].append((overlap_pred_gt[:,i] >= t).any())

    eval_result_NN_Rep = np.array(eval_result_NN_Rep).mean(axis=-1)
    for i in range(2):
        for j in range(len(tious)):
            eval_result_multilabel[i][j] = np.array(eval_result_multilabel[i][j]).mean(axis=-1)
    eval_result_multilabel = np.array(eval_result_multilabel)

    return eval_result_multilabel,eval_result_NN_Rep

def display_results(eval_result, miou, title=None):
    tious = [float(i) for i in config.TEST.TIOU.split(',')] if isinstance(config.TEST.TIOU,str) else [config.TEST.TIOU]
    recalls = [int(i) for i in config.TEST.RECALL.split(',')] if isinstance(config.TEST.RECALL,str) else [config.TEST.RECALL]

    display_data = [['Rank@{},mIoU@{}'.format(i,j) for i in recalls for j in tious]+['mIoU']]
    eval_result = eval_result*100
    miou = miou*100
    display_data.append(['{:.02f}'.format(eval_result[j][i]) for i in range(len(recalls)) for j in range(len(tious))]
                        +['{:.02f}'.format(miou)])
    table = AsciiTable(display_data, title)
    for i in range(len(tious)*len(recalls)):
        table.justify_columns[i] = 'center'
    return table.table

def display_results_new(eval_result_multilabel,title=None):
    tious = [float(i) for i in config.TEST.TIOU.split(',')] if isinstance(config.TEST.TIOU,str) else [config.TEST.TIOU]
    recalls = [int(i) for i in config.TEST.RECALL.split(',')] if isinstance(config.TEST.RECALL,str) else [config.TEST.RECALL]
    beta = config.TEST.EVAL_BETA

    ##our proposed multi-label metrics
    display_data=[['R@(5,5),IoU={}'.format(j) for j in tious]+['R_{}@(5,5),IoU={}'.format(beta,j) for j in tious]]
    MultiLabel_result = eval_result_multilabel*100
    display_data.append(['{:.02f}'.format(MultiLabel_result[i][j]) for i in range(2) for j in range(len(tious))])

    multi_table = AsciiTable(display_data, title)
    multi_table.inner_row_border = True
    for i in range(len(tious)*len(recalls)):
        multi_table.justify_columns[i] = 'center'
    return multi_table.table