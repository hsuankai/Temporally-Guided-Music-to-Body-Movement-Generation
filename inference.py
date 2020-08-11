#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 17:44:11 2019

@author: gaussian
"""

import os
import pickle
import numpy as np
import scipy.signal
from argument import parse_args

import torch
import torch.nn as nn

from model.utils import sort_sequences
from metric import compute_pck, bowing_acc
from model.network import MovementNet
from sklearn.metrics.pairwise import cosine_similarity as cosine


"""Options"""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
g = '0'
gpu_ids = [0]

v_train = ['04']
vid = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
pieces = ['bachno2', 'bachno3', 'bee5', 'bee6-1', 'bee6-2',
         'elgar', 'flower', 'mend-1', 'mend-2', 'mend-3', 'mozmvt1', 'mozmvt3-1', 'mozmvt3-2', 'wind']

if not os.path.isdir('result/body_data/'):
    os.makedirs('result/body_data/')

args = parse_args()

Set_mse_std = []
Set_mse_hand_std = []
Set_pck_std = []
Set_bow_std = []
Set_cosine_std = []

Set_mse = []
Set_mse_hand = []
Set_pck = []
Set_bow = []
Set_bow_std = []
Set_bowx = []
Set_bowy = []
Set_bowz = []
Set_cosine = []

#v_acc = {}
#for v in vid:
#    if v != '04':
#        v_acc[v] = []
V_acc = []
for ids, p in enumerate(pieces):
    print('testing %s ...' %p)

    """Data"""
    if args.mel:
        with open('../../Data/align/single_person/frame_bar_level_900/test_mel/' + p + '.pkl', 'rb') as f:
            Data = pickle.load(f)
    else:
        with open('../../Data/align/single_person/frame_bar_level_900/test_mfcc/' + p + '.pkl', 'rb') as f:
            Data = pickle.load(f)

    max_train = Data['max_train']
    max_test = Data['max_test']
    max_test_bar = Data['max_test_bar']
    body_mean, body_std = Data['keypoints_mean'], Data['keypoints_std']
    if max_train > max_test_bar:
        max_len = max_train
    else:
        max_len = max_test_bar

    """Model"""
    checkpoint = torch.load(args.checkpoint + 'nowarmup/' + p + '.pth')
    movement_net = MovementNet(args.d_input, args.d_output_body, args.d_output_rh, args.d_model, args.n_block, args.n_unet, args.n_attn, args.n_head, args.max_len, args.dropout,
                                   args.pre_lnorm, args.attn_type)
    movement_net = nn.DataParallel(movement_net, device_ids=gpu_ids)
    movement_net.load_state_dict(checkpoint['model_state_dict']['movement_net'])
    movement_net = movement_net.module
    movement_net.eval()
#------------------------ START TESTING ---------------------------------------
    result = {}
    result['Piece'] = p
    with torch.no_grad():
        set_mse = []
        set_mse_hand = []
        set_pck_01 = []
        set_pck_02 = []
        set_bow = []
        set_bowx = []
        set_bowy = []
        set_bowz = []
        set_cosine = []
        for v in vid:
            result[v] = {}
            aud = Data[v]['aud']
            keypoints = Data[v]['keypoints']
            beat = Data[v]['beat']
            Length = Data[v]['Length']
            sample_frame = Data[v]['sample_frame']
            bar_time = Data[v]['bar_time']
            onset = Data[v]['onset']
            seq_len = aud.shape[1]

            X_test = torch.tensor(aud, dtype=torch.float32).cuda('cuda:' + g)
            y_test = torch.tensor(keypoints, dtype=torch.float32).cuda('cuda:' + g)
            lengths = X_test.size(1)
            lengths = torch.tensor(lengths).cuda('cuda:' + g)
            
#            X_test = X_test.unsqueeze(0)
            lengths = lengths.unsqueeze(0)
            
            full_output = movement_net.forward(X_test, lengths)
                
            pred = full_output.squeeze(0)
            pred = pred.data.cpu().numpy()
            pred = pred[args.delay:]

            targ = keypoints.squeeze(0)
            targ = targ[args.delay:]

            assert pred.shape==targ.shape
            ''''''
            pred = pred * body_std + body_mean
            pred = np.reshape(pred, [len(pred), -1, 3])
            targ = targ * body_std + body_mean
            targ = np.reshape(targ, [len(targ), -1, 3])
            ''''''
            sample_time = sample_frame[int(args.delay/2):]
            assert len(pred)==len(sample_frame)
            sample_time = sample_time / 30
            sample_time = [sample_time[0], sample_time[-1]]
            
            pred_ = pred.reshape(len(pred), -1)
            pred_ = np.array([scipy.signal.medfilt(pred_[:, i], 5) for i in range(pred_.shape[1])], dtype='float32').transpose(1, 0).reshape(len(pred_), -1, 3)
            pred[:, :-2] = pred_[:, :-2]
#            pred = pred_
            
            result[v]['pred'] = pred
            result[v]['targ'] = targ
            result[v]['sample_time'] = sample_time

            if v not in v_train:

                piece_mse = np.mean(abs(pred - targ))
                piece_mse_hand = np.mean(abs(pred[:, -1, :] - targ[:, -1, :]))
                piece_pck_01 = compute_pck(pred, targ, alpha=0.1)
                piece_pck_02 = compute_pck(pred, targ, alpha=0.2)
                piece_bow_acc = bowing_acc(pred[:,-1,:], targ[:,-1,:], alpha=3)
                piece_cosine = np.mean(cosine(pred[:, -1, :], targ[:, -1, :]))
#                piece_cosine = []    
#                piece_cosine.append(np.mean(cosine(pred[:, -1, :], targ[:, -1, :])))
#                piece_cosine.append(np.mean(cosine(pred[:, -2, :], targ[:, -2, :])))
#                piece_cosine = np.mean(piece_cosine)
#                print(p + ' loss: %f' %piece_mse)
                set_mse.append(piece_mse)
                set_mse_hand.append(piece_mse_hand)
                set_pck_01.append(piece_pck_01)
                set_pck_02.append(piece_pck_02)
                set_bowx.append(piece_bow_acc[0])
                set_bowy.append(piece_bow_acc[1])
                set_bowz.append(piece_bow_acc[2])
                set_bow.append(piece_bow_acc[3])
                set_cosine.append(piece_cosine)
                
                V_acc.append(piece_bow_acc[3])

            torch.cuda.empty_cache()

        avg_pck = (np.mean(set_pck_01) + np.mean(set_pck_02))*0.5
        print(p + ' avg_mse_loss: %f' %np.mean(set_mse))
        print(p + ' avg_mse_hand_loss: %f' %np.mean(set_mse_hand))
        print(p + ' avg_pck_0.1: %f' %np.mean(set_pck_01))
        print(p + ' avg_pck_0.2: %f' %np.mean(set_pck_02))
        print(p + ' avg_pck: %f' %avg_pck)
        print(p + ' avg_bowing_accx: %f' %np.mean(set_bowx))
        print(p + ' avg_bowing_accy: %f' %np.mean(set_bowy))
        print(p + ' avg_bowing_accz: %f' %np.mean(set_bowz))
        print(p + ' avg_bowing_acc: %f' %np.mean(set_bow))
        print(p + ' avg_cosine_similarity: %f' %np.mean(set_cosine))
        Set_mse.append(np.mean(set_mse))
        Set_mse_hand.append(np.mean(set_mse_hand))
        Set_pck.append(avg_pck)
        Set_bow.append(np.mean(set_bow))
        Set_bow_std.append(np.std(set_bow))
        Set_bowx.append(np.mean(set_bowx))
        Set_bowy.append(np.mean(set_bowy))
        Set_bowz.append(np.mean(set_bowz))
        Set_cosine.append(np.mean(set_cosine))
        
        Set_mse_std.append(np.std(set_mse))
        Set_mse_hand_std.append(np.std(set_mse_hand))
        Set_pck_std.append((np.std(set_pck_01) + np.std(set_pck_02))*0.5)
        Set_bow_std.append(np.std(set_bow))
        Set_cosine_std.append(np.std(set_cosine))

        with open('result/body_data/nowarmup/' + p + '.pkl', 'wb') as f:
            pickle.dump(result, f)

print('\navg_mse_loss: %f' %(np.mean(Set_mse)))
print('avg_mse_hand_loss: %f' %(np.mean(Set_mse_hand)))
print('avg_pck: %f' %np.mean(Set_pck))
print('avg_bowing_accx: %f' %np.mean(Set_bowx))
print('avg_bowing_accy: %f' %np.mean(Set_bowy))
print('avg_bowing_accz: %f' %np.mean(Set_bowz))
print('avg_bowing_acc: %f' %np.mean(Set_bow))
print('avg_bowing_acc_std: %f' %np.mean(Set_bow_std))
print('avg_cosine_similarity: %f' %np.mean(Set_cosine))

print('\navg_mse_loss_std: %f' %(np.mean(Set_mse_std)*100))
print('avg_mse_hand_loss_std: %f' %(np.mean(Set_mse_hand_std)*10))
print('avg_pck: %f' %np.mean(Set_pck_std))
print('avg_bowing_acc_std: %f' %np.mean(Set_bow_std))
print('avg_cosine_similarity: %f' %np.mean(Set_cosine_std))

V_acc = np.array(V_acc)
np.save('v_fold_our', V_acc)
