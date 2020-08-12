#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 14:07:14 2020

@author: gaussian
"""

import os
import gdown

def download_data():  
    dst = 'data/'
    if not os.path.exists(dst):
        os.makedirs(dst)
    train_url = 'https://drive.google.com/uc?id=1TeENAdAiyIEqyPCjnedGB1Bq6POxF5us&export=download'
    test_url = 'https://drive.google.com/u/0/uc?id=1Nru_W7g65B-f1c7fMSRnj9q-iTycHHj-&export=download'
    train_dst = dst + 'train.pkl'
    test_dst = dst + 'test.pkl'
    gdown.download(train_url, train_dst)
    gdown.download(test_url, test_dst)

def download_checkpoint(dst):  
    checkpoint_url = 'https://drive.google.com/u/0/uc?id=1FhTp43x0vCxGqgxh5RDNCXPZ7_SfMh3E&export=download'
    gdown.download(checkpoint_url, dst)

