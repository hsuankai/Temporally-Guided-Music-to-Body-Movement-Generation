#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 16:32:01 2019

@author: gaussian
"""

import math
import numpy as np
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

class audio_skeleton_dataset(Dataset):
    def __init__(self, root, split, gpu='0'):
        ##############################################
        ### Initialize paths, transforms, and so on
        ##############################################
        with open(root, 'rb') as f:
            self.Data = pickle.load(f)
            
        self.data = self.Data[split]
        self.aud = self.data['aud']
        self.keypoints = self.data['keypoints']
        self.seq_len = self.data['seq_len']

        self.gpu = gpu
        
    def __getitem__(self, index):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        ##############################################
        aud = self.aud[index]
        keypoints = self.keypoints[index]
        seq_len = self.seq_len[index]
        
        aud = torch.tensor(aud, dtype=torch.float32).cuda('cuda:' + self.gpu)
        keypoints = torch.tensor(keypoints, dtype=torch.float32).cuda('cuda:' + self.gpu)
        seq_len = torch.tensor(seq_len).cuda('cuda:' + self.gpu)
        
        return aud, keypoints, seq_len

    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return len(self.aud)

   
def sort_sequences(inputs, lengths):
    """sort_sequences
    Sort sequences according to lengths descendingly.

    :param inputs (Tensor): input sequences, size [B, T, D]
    :param lengths (Tensor): length of each sequence, size [B]
    """
    lengths_sorted, sorted_idx = lengths.sort(descending=True)
    
    return inputs[sorted_idx], lengths_sorted


def delayArray(arr, delay, seq_len, dummy_var=0):
    arr[:, delay:, :] = arr[:, :(seq_len - delay):, :]
    arr[:, :delay, :] = dummy_var
    return arr

def lengths_to_tensor(lengths, max_len=265, gpu='0'):
    x_pos = [torch.arange(1, s+1).type(torch.int32) for s in lengths]
    x_pos = torch.stack([torch.cat((pos, torch.zeros(max_len - len(pos)).type(torch.int32))) for pos in x_pos]).cuda()
    return x_pos

def lambda_rule(epoch):
    lr_l = 1.0 - max(0, epoch + 1 - 100) / float(100 + 1)
    return lr_l
