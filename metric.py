# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 00:51:42 2019

@author: hsuankai
"""

import torch
import numpy as np
from sklearn.metrics import f1_score


def L1_loss(pred, target, mask):
    square_diff = torch.abs(pred - target)
    out = torch.sum(square_diff, 2, keepdim=True)
    masked_out = out * mask
    return torch.mean(masked_out)

def compute_pck(pred, gt, alpha=0.1):
    """
    https://github.com/amirbar/speech2gesture/blob/master/common/evaluation.py
    
    Args:
        pred: predicted keypoints on NxMxK where N is number of samples, M is of shape 2, corresponding to X,Y and K is the number of keypoints to be evaluated on
        gt:  similarly
        lpha: parameters controlling the scale of the region around the image multiplied by the max(H,W) of the person in the image. We follow https://www.cs.cmu.edu/~deva/papers/pose_pami.pdf and set it to 0.1
    Returns: 
        mean prediction score
    """
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

def compute_bowing_attack(direction):
    direction = np.sign(direction) # 1 0 -1
    temp = direction[0]
    bowing_attack = np.zeros_like(direction)
    for i, d in enumerate(direction):
        if i != 0:
            if d != temp:
                bowing_attack[i] = 1
                temp = d
    bowing_attack = bowing_attack.astype(np.int)
    return bowing_attack

def bowing_acc(pred, targ, alpha=3):
    """
    Args:
        pred, targ: [N, T, (K*3)]
        alpha: tolerance
    Returns:
        F1 score of bowing attack accuracy on x, y, z
    """
    F1 = []
    pred_direction = pred[1:] - pred[:-1]
    targ_direction = targ[1:] - targ[:-1]
    for coordinate in range(3):      
        pred_bow = compute_bowing_attack(pred_direction[:, coordinate]) 
        targ_bow = compute_bowing_attack(targ_direction[:, coordinate])
        prediction = []
        label = []
        i = 0
        index = np.zeros_like(targ_bow) # record which index has been calculated already
        for p, t in zip(pred_bow, targ_bow):
            if p == 1:
                prediction.append(1)
                if i-alpha < 0:
                    temp = targ_bow[0:i+alpha+1]
                    temp_idx = index[0:i+alpha+1]
                    
                elif i+alpha > len(pred_bow)-1:
                    temp = targ_bow[i-alpha:]
                    temp_idx = index[i-alpha:]
                    
                else:
                    temp = targ_bow[i-alpha:i+alpha+1]
                    temp_idx = index[i-alpha:i+alpha+1]
                    
                if 1 in temp:
                    idx = 0
                    token = 0
                    for t, t_idx in zip(temp, temp_idx):
                        if t == 1 and t_idx == 0:
                            label.append(1)
                            if i-alpha < 0:
                                index[idx] = 1
                                
                            else:
                                index[i-alpha+idx] = 1
                                
                            token = 1
                            break
                        idx += 1
                    if token == 0:
                        label.append(0)
                    
                else:
                    label.append(0)

            elif p==0 and t==1:
                prediction.append(0)
                label.append(1)
                
            elif p==0 and t==0:
                prediction.append(0)
                label.append(0)         
                
            i+=1             
            
        f1 = f1_score(label, prediction) 
        F1.append(f1)
    return (F1[0], F1[1], F1[2], np.mean(F1))
