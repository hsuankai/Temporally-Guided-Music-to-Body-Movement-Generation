#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 17:44:11 2019

@author: hsuankai
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import Download, audio_skeleton_dataset
from argument import parse
from metric import L1_loss
from model.utils import sort_sequences
from model.network import MovementNet
from model.optimizer import Optimizer


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

parser = parse()
args = parser.parse_args()

# Device
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpu_ids = [int(i) for i in args.gpu_ids.split(',')]


def main():
    # Data
    download_data = Download()
    download_data.train_data()
    train_dataset = audio_skeleton_dataset(download_data.train_dst, 'train', gpu_id=str(gpu_ids[0]))
    val_dataset = audio_skeleton_dataset(download_data.train_dst, 'val', gpu_id=str(gpu_ids[0]))

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)

    # Model
    movement_net = MovementNet(args.d_input, args.d_output_body, args.d_output_rh, args.d_model, args.n_block, args.n_unet, args.n_attn, args.n_head, args.max_len, args.dropout,
                                   args.pre_lnorm, args.attn_type)
    movement_net = nn.DataParallel(movement_net, device_ids=gpu_ids)
    optimizer = Optimizer(
        torch.optim.Adam(movement_net.parameters(), betas=(0.9, 0.98), eps=1e-09),
        1.0,
        args.d_model,
        args.warmup_steps)

    #------------------------ START TRAINING ---------------------------------#
    print('Training... \n' )
    if args.early_stop_iter > 0:
        counter = 0
    min_val_loss = float('inf')

    Epoch_train_loss = []
    Epoch_val_loss = []
    for e in range(args.epoch):
        print("epoch %d" %(e+1))

        # Training stage
        movement_net.train()

        pose_loss = []
        for X_train, y_train, seq_len in train_loader:

            X_train, lengths = sort_sequences(X_train, seq_len)
            y_train, _ = sort_sequences(y_train, seq_len)
            mask = y_train != 0
            mask = mask.type('torch.FloatTensor').cuda('cuda:' + str(gpu_ids[0]))

            full_output = movement_net.forward(X_train, lengths)

            loss = L1_loss(full_output, y_train, mask[:, :, :1])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(movement_net.parameters(), 1.)
            optimizer.step()

            pose_loss.append(loss.data.cpu().numpy())

        Epoch_train_loss.append(np.mean(pose_loss))
        print('train loss: ' + str(np.mean(pose_loss)))

        # Validation stage
        movement_net.eval()

        pose_loss = []
        with torch.no_grad():
            for X_val, y_val, seq_len in val_loader:

                X_val, lengths = sort_sequences(X_val, seq_len)
                y_val, _ = sort_sequences(y_val, seq_len)
                mask = y_val != 0
                mask = mask.type('torch.FloatTensor').cuda('cuda:' + str(gpu_ids[0]))

                full_output = movement_net.forward(X_val, lengths)

                loss = L1_loss(full_output, y_val, mask[:, :, :1])
                pose_loss.append(loss.data.cpu().numpy())

            Epoch_val_loss.append(np.mean(pose_loss))
            print('val loss: ' + str(np.mean(pose_loss)) + '\n')

            if counter == args.early_stop_iter:
                print("------------------early stopping------------------\n")
                break
            else:
                if min_val_loss > np.mean(pose_loss):
                    min_val_loss = np.mean(pose_loss)
                    counter = 0
                    if not os.path.exists('checkpoint'):
                        os.makedirs('checkpoint')
                    torch.save({'epoch' : e+1,
                                'model_state_dict': {'movement_net': movement_net.state_dict()},
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': min_val_loss}, args.checkpoint)
                else:
                    counter += 1

if __name__ == '__main__':
    main()