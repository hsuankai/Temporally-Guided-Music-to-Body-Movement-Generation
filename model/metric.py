# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 00:51:42 2019

@author: shiuan
"""

import torch
import numpy as np
from sklearn.metrics import f1_score

def build_loss(pred, target, mask):
    square_diff = (pred - target)**2
    out = torch.sum(square_diff, 2, keepdim=True)
    masked_out = out * mask
    return torch.mean(masked_out)

def build_loss_bow(pred, target, mask):
    square_diff = (pred - target)
    out = torch.sum(square_diff, 2, keepdim=True)
    masked_out = out * mask
    return torch.mean(masked_out)

def build_lossL1(pred, target, mask):
    square_diff = torch.abs(pred - target)
    out = torch.sum(square_diff, 2, keepdim=True)
    masked_out = out * mask
    return torch.mean(masked_out)

def compute_pck(pred, gt, alpha=0.1):
    '''
    :param pred: predicted keypoints on NxMxK where N is number of samples, M is of shape 2, corresponding to X,Y and K is the number of keypoints to be evaluated on
    :param gt:  similarly
    :param alpha: parameters controlling the scale of the region around the image multiplied by the max(H,W) of the person in the image. We follow https://www.cs.cmu.edu/~deva/papers/pose_pami.pdf and set it to 0.1
    :return: mean prediction score
    '''
    pred = np.reshape(pred, [len(pred), 3, -1])  
    gt = np.reshape(gt, [len(pred), 3, -1])
    pck_radius = compute_pck_radius(gt, alpha)
    keypoint_overlap = (np.linalg.norm(np.transpose(gt-pred, [0, 2, 1]), axis=2) <= (pck_radius))
    return np.mean(keypoint_overlap)

def compute_pck_radius(gt, alpha):
    width = np.abs(np.max(gt[:, 0:1], axis=2) - np.min(gt[:, 0:1], axis=2))
    depth = np.abs(np.max(gt[:, 1:2], axis=2) - np.min(gt[:, 1:2], axis=2))
    height = np.abs(np.max(gt[:, 2:3], axis=2) - np.min(gt[:, 2:3], axis=2))
    max_axis = np.concatenate([width, depth, height], axis=1).max(axis=1)
    max_axis_per_keypoint = np.tile(np.expand_dims(max_axis, -1), [1, 15])
    return max_axis_per_keypoint * alpha



def gaussian(x, std):
     return torch.exp((-(x) ** 2)/(2 * std))
 
def cal_changing_pt_pred(pred, std=0.1):
    
    direction = pred[:, 1:] - pred[:, :-1]
#    direction = direction / (torch.abs(direction + 1E-9)) # normalize to -1 0 1
#    direction = direction / torch.abs(direction + 1E-9)**0.5 # normalize to -1 0 1
    change_pts = gaussian(direction, std)
#    change_pts = direction[:, 1:] - direction[:, :-1] + 1E-9 # find changing pts
#    change_pts = torch.abs(change_pts) # -2 -> 2 2 -> 2
#    change_pts = torch.tanh(change_pts) # 2 -> 0.96 # 1E-9 -> 0.~
    
    return change_pts

#def cal_changing_pt_targ(pred, std=0.1):
#    
#    direction = pred[:, 1:] - pred[:, :-1]
##    direction = direction / (torch.abs(direction + 1E-9)) # normalize to -1 0 1
##    direction = direction / torch.abs(direction + 1E-9)**0.5
#    change_pts = gaussian(direction, std)
##    change_pts = direction[1:] - direction[:-1] + 1E-9 # find changing pts
##    change_pts = torch.abs(change_pts) # -2 -> 2 2 -> 2
##    change_pts = torch.tanh(change_pts) # 2 -> 0.96 # 1E-9 -> 0.~
#    
#    return change_pts

def cal_changing_pt_targ_torch(targ):
    
    targ = targ.data.cpu().numpy()
#    pred_direction = targ[:, 1:] - targ[:, :-1]
    pred_direction = np.sign(targ)
    
    bow_attack = []
    for p in pred_direction:
        temp = p[0]
        pred_bow = np.zeros_like(p)
        for i, direct in enumerate(p):
            if i!=0:
                if direct!=temp:
                    pred_bow[i] = 1
                    temp = direct
        
        assert len(p) == len(pred_bow)
        pred_bow = pred_bow.astype(np.int)
        bow_attack.append(pred_bow)
    bow_attack = np.array(bow_attack)
    bow_attack = torch.tensor(bow_attack, dtype=torch.float32).cuda()
#    bow_attack = bow_attack.unsqueeze(-1)

    return bow_attack

def cal_changing_pt_targ(pts):
    direction = pts[1:] - pts[:-1] > 0
    temp = direction[0]
    changing_pt = np.zeros_like(direction)
    for i, direct in enumerate(direction):
        if i!=0:
            if direct!=temp:
                changing_pt[i] = 1
                temp = direct
    
    assert len(pts)-1 == len(changing_pt)
    changing_pt = changing_pt.astype(np.int)
    return changing_pt

def gaussian_np(x, std):
     return np.exp((-(x) ** 2)/(2 * std))

def cal_changing_pt_pred_np(pred_direction):
    pred_direction = np.sign(pred_direction)
    temp = pred_direction[0]
    pred_bow = np.zeros_like(pred_direction)
    for i, direct in enumerate(pred_direction):
        if i!=0:
            if direct!=temp:
                pred_bow[i] = 1
                temp = direct
    
    assert len(pred_direction) == len(pred_bow)
    pred_bow = pred_bow.astype(np.int)

    
    return pred_bow

from itertools import groupby
from operator import itemgetter

def bow_attack_idx(direction):    
    f = lambda x: x[0] - x[1]
    idx = []
    for k, g in groupby(enumerate(direction), f):
        idx.append(list(map(itemgetter(1), g))[-1])
    idx = np.array(idx, dtype='int32')
    return idx

def cal_changing_pt_targ_np(targ_bow):

    h = targ_bow
    bow_attack = np.zeros_like(h)
    idx = np.where(h==1)[0]
    if len(idx)!=0:
#        idx = bow_attack_idx(idx)

#        idx_1 = idx - 1
#        idx_2 = idx - 2
        bow_attack[idx] = 1
#        bow_attack[idx_1] = 1
#        bow_attack[idx_2] = 1

#    bow_attack_label = np.zeros_like(bow_attack)
#    idx = np.where(bow_attack==1)[0]
#    idx = bow_attack_idx(idx)
#    bow_attack_label[idx] = 1

    return bow_attack

def bowing_acc(pred, targ, alpha=1):
    F1 = []
    pred_direction = pred[1:] - pred[:-1]
    targ_direction = targ[1:] - targ[:-1]
#    predict_bow = gaussian_np(pred_direction, std=1E-9)
#    target_bow = gaussian_np(targ_direction, std=1E-9)
    
    for pt in range(3):      
#        pred_bow = cal_changing_pt_targ_np(predict_bow[:, pt])
        pred_bow = cal_changing_pt_pred_np(pred_direction[:, pt])
#        targ_bow = target_bow[:, pt]
#        targ_bow = cal_changing_pt_targ_np(target_bow[:, pt])
        targ_bow = cal_changing_pt_pred_np(targ_direction[:, pt])
        prediction = []
        label = []
        i = 0
        index = np.zeros_like(targ_bow)
        for p, t in zip(pred_bow, targ_bow):
            if p==1:
                prediction.append(1)
                if i-alpha<0:
                    temp = targ_bow[0:i+alpha+1]
                    temp_idx = index[0:i+alpha+1]
                elif i+alpha>len(pred_bow)-1:
                    temp = targ_bow[i-alpha:]
                    temp_idx = index[i-alpha:]
                else:
                    temp = targ_bow[i-alpha:i+alpha+1]
                    temp_idx = index[i-alpha:i+alpha+1]
                    
                if 1 in temp:
                    idx = 0
                    token = 0
                    for t, t_idx in zip(temp, temp_idx):
                        if t==1 and t_idx==0:
                            label.append(1)
                            index[i-alpha+idx] = 1
                            token = 1
                            break
                        idx += 1
                    if token==0:
                        label.append(0)
                    
                else:
                    label.append(0)

            elif t==1:
                label.append(1)
                prediction.append(0)
                    
            elif p==0 and t==0:
                prediction.append(0)
                label.append(0)
            
            i+=1
                        
        f1 = f1_score(label, prediction) 
        F1.append(f1)
        
    return (F1[0], F1[1], F1[2], np.mean(F1))
