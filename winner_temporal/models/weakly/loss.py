import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from runners.main_runner import calculate_IoU_batch3

def sigmoid_and_normalize(scores):
    joint_prob = torch.sigmoid(scores)
    min_prob, max_prob = joint_prob.min(dim=-1, keepdim=True)[0], \
                         joint_prob.max(dim=-1, keepdim=True)[0]
    joint_prob_norm = (joint_prob - min_prob + 1e-10) / (max_prob - min_prob + 1e-10)
    return joint_prob, joint_prob_norm


def bce_rescale_loss(scores, targets, min_iou=0.5, max_iou=1.0, bias=0.0, reduction='mean'):
    joint_prob, joint_prob_norm = sigmoid_and_normalize(scores)
    joint_prob = joint_prob_norm

    target_prob = (targets - min_iou) * (1 - bias) / (max_iou - min_iou)
    target_prob[target_prob > 0] += bias
    target_prob[target_prob > 1] = 1
    target_prob[target_prob < 0] = 0

    # print("\n")
    # print(joint_prob_norm)

    loss = F.binary_cross_entropy(joint_prob_norm, target_prob, reduction='none')
    loss_value = loss.mean(dim=-1)
    if reduction == 'mean':
        loss_value = loss_value.mean(dim=-1)
    elif reduction == 'none':
        loss_value = loss_value
    else:
        loss_value = loss_value.sum(dim=-1)
    return loss_value, joint_prob

def supervised_loss(score, map_gt, loss_meter=None,
                    reduction="mean", min_iou=0.5, max_iou=1.0,
                    delta=None, probs=None, gt_box=None, durations=None):
    
    joint_prob = torch.sigmoid(score)
    # softmax = nn.Softmax(dim=1)
    # joint_prob = softmax(score)
    
    target_prob = map_gt
    target_prob = (map_gt - min_iou)/(max_iou - min_iou)
    target_prob[target_prob>1] = 1
    target_prob[target_prob<0] = 0

    align_func = nn.BCELoss()
    loss_align = align_func(joint_prob, target_prob)
    
    indices = torch.argmax(score, -1)
    gt_indices = torch.argmax(map_gt, -1)
    
    predict_box = probs[indices]
    predict_box = predict_box.cuda()
    gt_box = gt_box * 64 / durations[:,None]
    
    predict_reg = delta[range(delta.size(0)), indices]
    
    refine_box = predict_box + predict_reg
    
    reg_func = nn.SmoothL1Loss()
    reg_loss = reg_func(refine_box, gt_box)
 
    super_loss = loss_align + 0.01 * reg_loss
    
    loss_meter['super_loss'].update(super_loss)
    # print("super_loss is ", super_loss)
    return super_loss

    # previous version
    # lambda_loss = 0.5
    # softmax_probs = F.softmax(score, dim=1)
    
    # norm_probs, _ = sigmoid_and_normalize(score)
    
    # target_prob = copy.deepcopy(map_gt)
    # target_prob = (target_prob - min_iou) / (max_iou - min_iou)
    # target_prob[target_prob < 0.0] = 0
    # target_prob[target_prob > 1] = 1

    # super_loss = F.binary_cross_entropy(softmax_probs, target_prob, reduction='none')
    # loss_period = super_loss.mean(dim=-1)
    # if reduction == "mean":
    #     loss_period = loss_period.mean(dim=-1)
    
    # batch_size = norm_probs.size(0)
    # # norm_probs = norm_probs.reshape(batch_size, -1)
    # gt_index = torch.argmax(map_gt, dim=1).reshape(-1)
    
    # tmp = torch.arange(batch_size)
    # inputs = softmax_probs[tmp, gt_index[tmp]]

    # loss_fuc = torch.nn.BCELoss()
    # targets = torch.ones([batch_size]).cuda()
    # loss_find_max = loss_fuc(inputs, targets)
    # loss_value = loss_find_max * lambda_loss + loss_period * lambda_loss
    # # loss_value = loss_period * lambda_loss
    # loss_meter["super_loss"].update(loss_value)
    # return loss_value

    # print(norm_probs)
    # gt_index = torch.argmax(map_gt, dim=1).reshape(-1)
    # # print(gt_index)
    # tmp = torch.arange(batch_size)
    # inputs = norm_probs[tmp, gt_index[tmp]]
    # # print(inputs)

    # loss_fuc = torch.nn.BCELoss()
    # targets = torch.ones([batch_size]).cuda()
    # super_loss = loss_fuc(inputs, targets)
    
    # loss_meter['super_loss'].update(super_loss)
    # # print("super_loss is ", super_loss)
    # return super_loss


def weakly_supervised_loss(pos_score, neg_score1, neg_score2, neg_weight2, weight_gt, props,
                           num_cands, log_fp=None, loss_meter=None, **kwargs):
    info = ''

    def calc_loss(score, positive=True):
        bsz, num_clips = score.size()
        joint_prob, joint_prob_norm = sigmoid_and_normalize(score)

        # joint_prob = joint_prob_norm

        idx = torch.argsort(joint_prob_norm, dim=-1, descending=True)
        props1 = props[idx[:, :num_cands]].contiguous()  # [bsz, 200, 2]
        props2 = props1[:, 0]
        props2 = props2.unsqueeze(1).expand(bsz, num_cands, 2).contiguous().view(bsz * num_cands, 2)
        props1 = props1.view(bsz * num_cands, 2)
        iou = calculate_IoU_batch((props2[:, 0], props2[:, 1]), (props1[:, 0], props1[:, 1]))
        iou = iou.contiguous().view(bsz, num_cands)
        iou = iou.type_as(joint_prob_norm)

        sort_idx = torch.argsort(iou, dim=-1, descending=True)[:, :kwargs['topK']]
        idx1 = idx.gather(dim=-1, index=sort_idx)
        tmp = joint_prob.gather(dim=-1, index=idx1)
        # tmp = joint_prob.gather(dim=-1, index=idx)[:, :kwargs['topK']]
        align_score = tmp.mean(dim=-1)

        # log_fp.write('{}, {}\n'.format(positive, tmp[0]))
        if positive:
            tmp1 = joint_prob_norm.mean()
            nonlocal info
            info += 'soc {}, '.format(float(tmp1))
            # tmp1 = F.relu(joint_prob_norm.mean(dim=-1) - 0.6).mean()
            # tmp2 = (tmp * F.log_softmax(tmp, -1)).mean()
            tmp2 = -(tmp.softmax(dim=-1) * tmp.log_softmax(dim=-1)).sum(dim=-1).mean()
            tmp2 = -(tmp.log_softmax(dim=-1).max(dim=-1)[0]).mean()
            # tmp2 = F.log_softmax(tmp, -1).mean()
            # charades: 5e-2, 1e-1
            # norm_loss = 1e-2 * tmp1 + 1e-2 * tmp2
            norm_loss = tmp1 + tmp2
        else:
            norm_loss = None

        return joint_prob_norm, align_score, norm_loss

    joint_prob_norm, pos_score, norm_loss1 = calc_loss(pos_score, positive=True)
 
    _, neg_score1, neg_norm_loss1 = calc_loss(neg_score1, positive=False)
    info += 'pos {}, neg1 {}, '.format(float(pos_score.mean(dim=-1)), float(neg_score1.mean(dim=-1)))
    # inter_loss = F.relu(neg_score1 - pos_score + 0.2).mean(dim=-1)
    inter_loss = (-torch.log(pos_score + 1e-10) + -torch.log(1.0 - neg_score1 + 1e-10)).mean()
    loss_meter['inter_loss'].update(F.relu(neg_score1 - pos_score + 0.2).mean(dim=-1).item())
    # print(pos_score.mean(), neg_score1.mean())
    # loss_meter['inter_loss'].update(inter_loss.item())
    loss_meter['norm_loss1'].update(norm_loss1.item())
    # print(norm_loss1)

    if neg_score2 is not None:
        _, neg_score2, _ = calc_loss(neg_score2, positive=False)
        # Charades: 1e-1, 1e-2, 1e-2
        # weight_max, weight_min = neg_weight2.max(dim=-1, keepdim=True)[0], neg_weight2.min(dim=-1, keepdim=True)[0]
        # weight_nrom = (neg_weight2 - weight_min + 1e-10) / (weight_max - weight_min + 1e-10)
        neg_weight2 = neg_weight2.squeeze(-1)
        intra_loss = F.relu(neg_score2 - pos_score + 0.2).mean(dim=-1)
        tmp = -(neg_weight2.softmax(dim=-1) * neg_weight2.log_softmax(dim=-1)).mean()
        # print()
        # print(neg_weight2[0].tolist())
        # exit(0)
        # print(neg_weight2.mean())
        # norm_loss2 = neg_weight2.mean() + tmp
        # print(neg_weight2[0])
        norm_loss2 = neg_weight2.mean() + tmp
        # weight_loss = F.binary_cross_entropy(neg_weight2, weight_gt)

        info += 'neg2 {}'.format(float(neg_score2.mean()))
        loss_meter['intra_loss'].update(intra_loss.item())
        loss_meter['norm_loss2'].update(norm_loss2.item())
        # loss_meter['weight_loss'].update(weight_loss.item())
    else:
        intra_loss = 0.0
        norm_loss2 = 0.0
    if log_fp is not None:
        log_fp.write(info + '\n')
        log_fp.flush()
    final_loss = inter_loss \
                 + kwargs['norm1'] * norm_loss1 \
                 + kwargs['intra'] * intra_loss \
                 + kwargs['norm2'] * norm_loss2
    # final_loss = inter_loss + 1e-2 * norm_loss1 + 1e-1 * intra_loss + 1e-2 * norm_loss2
    return final_loss, joint_prob_norm


def weakly_supervised_loss_fuck(pos_score, neg_score1, neg_score2, neg_weight2, weight_gt, props,
                           num_cands, log_fp=None, loss_meter=None, **kwargs):
    info = ''

    def calc_loss(score, positive=True):
        bsz, num_clips = score.size()
        joint_prob, joint_prob_norm = sigmoid_and_normalize(score)
        # print(score)
        # print(score.shape)
        # exit(0)
        # joint_prob = joint_prob_norm

        idx = torch.argsort(joint_prob_norm, dim=-1, descending=True)
        props1 = props[idx[:, :num_cands]].contiguous()  # [bsz, 200, 2]
        props2 = props1[:, 0]
        props2 = props2.unsqueeze(1).expand(bsz, num_cands, 2).contiguous().view(bsz * num_cands, 2)
        props1 = props1.view(bsz * num_cands, 2)


        iou = calculate_IoU_batch((props2[:, 0], props2[:, 1]), (props1[:, 0], props1[:, 1]))
        iou = iou.contiguous().view(bsz, num_cands)
        iou = iou.type_as(joint_prob_norm)

        sort_idx = torch.argsort(iou, dim=-1, descending=True)[:, :kwargs['topK']]
        idx1 = idx.gather(dim=-1, index=sort_idx)
        tmp = joint_prob.gather(dim=-1, index=idx1)
        # tmp = joint_prob.gather(dim=-1, index=idx)[:, :kwargs['topK']]
        align_score = tmp.mean(dim=-1)

        # log_fp.write('{}, {}\n'.format(positive, tmp[0]))
        if positive:
            tmp1 = joint_prob_norm.mean()
            nonlocal info
            info += 'soc {}, '.format(float(tmp1))
            # tmp1 = F.relu(joint_prob_norm.mean(dim=-1) - 0.6).mean()
            # tmp2 = (tmp * F.log_softmax(tmp, -1)).mean()
            tmp2 = -(tmp.softmax(dim=-1) * tmp.log_softmax(dim=-1)).sum(dim=-1).mean()
            tmp2 = -(tmp.log_softmax(dim=-1).max(dim=-1)[0]).mean()
            # tmp2 = F.log_softmax(tmp, -1).mean()
            # charades: 5e-2, 1e-1
            # norm_loss = 1e-2 * tmp1 + 1e-2 * tmp2
            norm_loss = tmp1 + tmp2
        else:
            norm_loss = None

        return joint_prob_norm, align_score, norm_loss

    joint_prob_norm, pos_score, norm_loss1 = calc_loss(pos_score, positive=True)

    _, neg_score2, _ = calc_loss(neg_score2, positive=False)
    # Charades: 1e-1, 1e-2, 1e-2
    # weight_max, weight_min = neg_weight2.max(dim=-1, keepdim=True)[0], neg_weight2.min(dim=-1, keepdim=True)[0]
    # weight_nrom = (neg_weight2 - weight_min + 1e-10) / (weight_max - weight_min + 1e-10)

    return neg_score2, pos_score


def calculate_IoU_batch(i0, i1):
    union = (torch.min(torch.stack([i0[0], i1[0]], 0), 0)[0], torch.max(torch.stack([i0[1], i1[1]], 0), 0)[0])
    inter = (torch.max(torch.stack([i0[0], i1[0]], 0), 0)[0], torch.min(torch.stack([i0[1], i1[1]], 0), 0)[0])
    iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


def calculate_IoU_batch_didemo(i0, i1):
    union = (torch.min(torch.stack([i0[0], i1[0]], 0), 0)[0], torch.max(torch.stack([i0[1], i1[1]], 0), 0)[0])
    inter = (torch.max(torch.stack([i0[0], i1[0]], 0), 0)[0], torch.min(torch.stack([i0[1], i1[1]], 0), 0)[0])
    iou = 1.0 * (inter[1] - inter[0] + 1.0) / (union[1] - union[0] + 1.0)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou
