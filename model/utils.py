#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 16:32:01 2019

@author: hsuankai
"""

import torch


def sort_sequences(inputs, seq_len):
    """
    Sort sequences according to lengths descendingly.

    inputs (Tensor): input sequences, size [B, T, D]
    seq_len (Tensor): length of each sequence, size [B]
    """
    lengths_sorted, sorted_idx = seq_len.sort(descending=True)
    
    return inputs[sorted_idx], lengths_sorted

def lengths_to_tensor(lengths, max_len=900, gpu='0'):
    x_pos = [torch.arange(1, s+1).type(torch.int32) for s in lengths]
    x_pos = torch.stack([torch.cat((pos, torch.zeros(max_len - len(pos)).type(torch.int32))) for pos in x_pos]).cuda()
    return x_pos

def delayArray(arr, delay, seq_len, dummy_var=0):
    arr[:, delay:, :] = arr[:, :(seq_len - delay):, :]
    arr[:, :delay, :] = dummy_var
    return arr