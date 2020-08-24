import os
import gdown
import pickle
import librosa
import numpy as np

import torch
from torch.utils.data.dataset import Dataset


class Download():
    def __init__(self):
        self.data_dst = 'data/'
        self.checkpoint_dst = 'checkpoint/'
        if not os.path.exists(self.data_dst):
            os.makedirs(self.data_dst)
        if not os.path.exists(self.checkpoint_dst):
            os.makedirs(self.checkpoint_dst)
        self.train_dst = self.data_dst + 'train.pkl'
        self.test_dst = self.data_dst + 'test.pkl'
        self.wav_dst = self.data_dst + 'flower.wav'
        self.pretrain_model_dst = self.checkpoint_dst + 'checkpoint081220.pth'
        self.train_url = 'https://drive.google.com/uc?id=1QsghRzGwgzZBQz03MqtWZ0S7X0Y6NivC&export=download'
        self.test_url = 'https://drive.google.com/u/0/uc?id=1WQksHdEH65xES557nkbsIuNM69vSdtYq&export=download'
        self.wav_url = 'https://drive.google.com/u/0/uc?id=1WwSMkhe5ga0GQdk9OC4atfVaAWkPNd3X&export=download'
        self.pretrain_model_url = 'https://drive.google.com/u/0/uc?id=1EMSo0M4ITkNK0Hkj72bRO0l0fejPA1jn&export=download'
    
    def train_data(self):
        if not os.path.exists(self.train_dst):
            gdown.download(self.train_url, self.train_dst)
    
    def test_data(self):
        if not os.path.exists(self.test_dst):
            gdown.download(self.test_url, self.test_dst)
        if not os.path.exists(self.wav_dst):
            gdown.download(self.wav_url, self.test_wav_dst)
            
    def pretrain_model(self):
        if not os.path.exists(self.pretrain_model_dst):
            gdown.download(self.pretrain_model_url, self.pretrain_model_dst)

class audio_skeleton_dataset(Dataset):
    """
        aud: MFCC feature, size [N, T, D]
        keypoints: skeleton feature, size [N, T, (K*3)]
        seq_len: length of each sequence, size [N] 
    """
    def __init__(self, root, split, gpu_id='0'):
        with open(root, 'rb') as f:
            self.Data = pickle.load(f)
            
        self.data = self.Data[split]
        self.aud = self.data['aud']
        self.keypoints = self.data['keypoints']
        self.seq_len = self.data['seq_len']

        self.gpu_id = gpu_id
        
    def __getitem__(self, index):
        aud = self.aud[index]
        keypoints = self.keypoints[index]
        seq_len = self.seq_len[index]
        
        aud = torch.tensor(aud, dtype=torch.float32).cuda('cuda:' + self.gpu_id)
        keypoints = torch.tensor(keypoints, dtype=torch.float32).cuda('cuda:' + self.gpu_id)
        seq_len = torch.tensor(seq_len).cuda('cuda:' + self.gpu_id)
        
        return aud, keypoints, seq_len

    def __len__(self):
        return len(self.aud)

def preprocess(audio, aud_mean, aud_std):
    """ Extract MFCC feature """
    n_fft = 4096
    hop = 1470
    y, sr = librosa.load(audio, sr=44100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mfcc=13)
    energy = np.log(librosa.feature.rmse(y=y, frame_length=n_fft, hop_length=hop, center=True))
    mfcc_energy = np.vstack((mfcc, energy))
    mfcc_delta = librosa.feature.delta(mfcc_energy)
    aud = np.vstack((mfcc_energy, mfcc_delta)).T
    aud = (aud - aud_mean) / (aud_std + 1E-8)
    return aud