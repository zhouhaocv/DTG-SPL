from core.config import config
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

def giou(pred, gt):
    eps = 1e-8
    gt = gt.clone().detach()
    inter_left = torch.max(pred[:,:,0,None], gt[:,None,:,0])
    inter_right = torch.min(pred[:,:,1,None], gt[:,None,:,1])
    inter = torch.clamp(inter_right - inter_left,0.0)
    union_left = torch.min(pred[:,:,0,None], gt[:,None,:,0])
    union_right = torch.max(pred[:,:,1,None], gt[:,None,:,1])
    union = torch.clamp(union_right - union_left,eps)
    inter2 = torch.clamp(inter_left - inter_right,0.0)

    overlap = (inter - inter2) / union
    return overlap

@torch.no_grad()   
def multi_moment_selection(outputs_se,outputs_cw, moment_anchors,moment_num):
    """ Performs the matching
    """
    score_start2 = outputs_cw[:,:,0] - 0.5*outputs_cw[:,:,1]
    score_end2 = outputs_cw[:,:,0] + 0.5*outputs_cw[:,:,1]
    outputs_cw_to_se = torch.stack((score_start2,score_end2),dim=2)
    outputs_se = outputs_se
    outputs_cw_to_se = outputs_cw_to_se
    moment_anchors = moment_anchors[:,:2].repeat(outputs_se.size(0),1,1)
    loss_se_giou = - giou(outputs_se,moment_anchors)
    loss_cw_giou = - giou(outputs_cw_to_se,moment_anchors)

    C = loss_se_giou + loss_cw_giou
    C = C.cpu()
    indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(moment_num, -1))]
    return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def reg_loss(prediction, mid_out,gt_info):
    gamma1 = config.LOSS.GAMMA1
    gamma2 = config.LOSS.GAMMA2
    lam = config.LOSS.LAMBDA
    epr_value = config.LOSS.EPR
    p_se,p_cw = prediction
    a_multi,rank_scores,caption_out,ret_dict = mid_out
    batch_size = p_se.size(0)

    moment_masks = gt_info['batch_moment_masks'].cuda()
    moment_anchors = gt_info['batch_moment_anchors'].cuda()
    moment_num = gt_info['batch_moment_num']
    gt_masks = gt_info['batch_gt_masks'].cuda()
    batch_gt_caption = gt_info['batch_gt_caption'].cuda()
    targets = gt_info['batch_map_gt'].cuda()
    gt_se = gt_info['batch_gt_time'].cuda()

    #multi branch top 1
    gt_center = torch.mean(gt_se,1)
    gt_width = gt_se[:,1]-gt_se[:,0]
    loss_single_se = F.l1_loss(p_se[:,0],gt_se)
    loss_single_cw = F.l1_loss(p_cw[:,0,0],gt_center) + F.l1_loss(p_cw[:,0,1],gt_width)
    loss_single_att = (-gt_masks*torch.log(a_multi[:,0]+1e-8)).sum(1)/gt_masks.sum(1)
    loss_single_att = loss_single_att.sum(0)/batch_size

    #multi_moment_matching
    indices = multi_moment_selection(p_se[:,1:],p_cw[:,1:],moment_anchors,moment_num)
    src_idx = get_src_permutation_idx(indices)
    ##select prediction
    p_se2 = p_se[:,1:][src_idx]
    p_cw2 = p_cw[:,1:][src_idx]
    a_multi2 = a_multi[:,1:][src_idx]

    ##select pseudo moments
    pseudo_mask = torch.cat([t[i] for t, (_, i) in zip(moment_masks.split(moment_num), indices)], dim=0)
    pseudo_se = torch.cat([t[i] for t, (_, i) in zip(moment_anchors.split(moment_num), indices)], dim=0)
    pseudo_center = torch.mean(pseudo_se,1)
    pseudo_width = pseudo_se[:,1]-pseudo_se[:,0]

    #regression branch
    loss_multi_se = F.l1_loss(p_se2,pseudo_se)
    loss_multi_cw = F.l1_loss(p_cw2[:,0],pseudo_center) + F.l1_loss(p_cw2[:,1],pseudo_width)
    loss_multi_att = (-pseudo_mask*torch.log(a_multi2+1e-8)).sum(-1)/pseudo_mask.sum(-1)
    loss_multi_att = loss_multi_att.mean(0)

    ##estimation branch
    masks = targets.clone()
    masks[targets >= 0.5] = 1.0
    masks[targets < 0.5] = 0.0
    loss_match_rank = F.binary_cross_entropy(rank_scores,targets,reduction='none') * masks
    loss_match_rank = torch.sum(loss_match_rank) / torch.sum(masks)
    batch_ranksize = rank_scores.sum()/rank_scores.size(0)
    loss_match_rank_epr = (batch_ranksize - epr_value)**2

    gt_caption = batch_gt_caption[:,1:].permute(1,0).contiguous().repeat(caption_out.size(2),1,1).view(caption_out.size(2),-1).view(-1).long()
    caption_out_gt = caption_out.permute(2,0,1,3).contiguous().view(caption_out.size(2),-1,caption_out.size(-1)).view(-1,caption_out.size(-1))
    loss_semantic = F.nll_loss(caption_out_gt,gt_caption)

    loss_multi = loss_multi_se + loss_multi_att + loss_multi_cw 
    loss_single = loss_single_se + loss_single_att + loss_single_cw
    loss_dmr = loss_single + lam*loss_multi
    loss_pme = loss_match_rank + gamma1 * loss_match_rank_epr + gamma2 * loss_semantic
    loss_all = loss_pme + loss_dmr

    return loss_all