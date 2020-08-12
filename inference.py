#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 17:44:11 2019

@author: gaussian
"""

import os
import pickle
import librosa
import numpy as np
import scipy.signal
from argument import parse_args

import torch
import torch.nn as nn

from model.network import MovementNet
from visualize import render_animation
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from download import download_checkpoint


def plot(audio_path, output_path, pred):
    render_animation("", fps=30, output_path='temp.mp4', azim=75, prediction=pred, knee=True)
    audioclip = AudioFileClip(audio_path, fps=44100)
    videoclip = VideoFileClip('temp.mp4')
    videoclip.audio = audioclip
    videoclip.write_videofile(output_path, fps=30)


"""Options"""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
g = '0'
gpu_ids = [0]



args = parse_args()

n_fft = 4096
hop = 1470
y, sr = librosa.load(args.inference_data, sr=44100)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mfcc=13)
energy = np.log(librosa.feature.rmse(y=y, frame_length=n_fft, hop_length=hop, center=True))
mfcc_energy = np.vstack((mfcc, energy))
mfcc_delta = librosa.feature.delta(mfcc_energy)
aud = np.vstack((mfcc_energy, mfcc_delta)).T


print('inference...')
if not os.path.exists('checkpoint'):
    os.makedirs('checkpoint')
if not os.path.exists(args.pretrain):
    download_checkpoint(args.pretrain)

checkpoint = torch.load(args.pretrain)
keypoints_mean, keypoints_std = checkpoint['keypoints_mean'], checkpoint['keypoints_std']
aud_mean, aud_std = checkpoint['aud_mean'], checkpoint['aud_std']
aud = (aud - aud_mean) / (aud_std + 1E-8)

"""Model"""

movement_net = MovementNet(args.d_input, args.d_output_body, args.d_output_rh, args.d_model, args.n_block, args.n_unet, args.n_attn, args.n_head, args.max_len, args.dropout,
                               args.pre_lnorm, args.attn_type)
movement_net = nn.DataParallel(movement_net, device_ids=gpu_ids)
movement_net.load_state_dict(checkpoint['model_state_dict']['movement_net'])
movement_net = movement_net.module
movement_net.eval()
#------------------------ START TESTING ---------------------------------------
result = {}

with torch.no_grad():
    seq_len = len(aud)

    X_test = torch.tensor(aud, dtype=torch.float32).cuda('cuda:' + g).unsqueeze(0)
    lengths = X_test.size(1)
    lengths = torch.tensor(lengths).cuda('cuda:' + g)
    lengths = lengths.unsqueeze(0)
    
    full_output = movement_net.forward(X_test, lengths)
        
    pred = full_output.squeeze(0)
    pred = pred.data.cpu().numpy()
    pred = pred[args.delay:]

    ''''''
    pred = pred * keypoints_std + keypoints_mean
    pred = np.reshape(pred, [len(pred), -1, 3])
    ''''''
    
    pred_ = pred.reshape(len(pred), -1)
    pred_ = np.array([scipy.signal.medfilt(pred_[:, i], 5) for i in range(pred_.shape[1])], dtype='float32').transpose(1, 0).reshape(len(pred_), -1, 3)
    pred[:, :-2] = pred_[:, :-2]

    torch.cuda.empty_cache()

plot(args.inference_data, args.animation_output, pred)



