import os
import pickle
import numpy as np
from argument import parse

import torch
import torch.nn as nn

from data import Download, preprocess
from model.network import MovementNet
from visualize.animation import plot


def main():
    # Parser
    parser = parse()
    parser.add_argument('--inference_audio', type=str, default='inference.wav', help='the path of input wav file', required=True)
    parser.add_argument('--plot_path', type=str, default='inference.mp4', help='plot skeleton and add audio')
    parser.add_argument('--output_path', type=str, default='inference.pkl', help='save skeletal data')
    args = parser.parse_args()
    
    # Device
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    # Load pretrain model
    download_data = Download()
    download_data.pretrain_model()
    checkpoint = torch.load(download_data.pretrain_model_dst, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
    keypoints_mean, keypoints_std = checkpoint['keypoints_mean'], checkpoint['keypoints_std']
    aud_mean, aud_std = checkpoint['aud_mean'], checkpoint['aud_std']
    
    # Audio pre-processing
    aud = preprocess(args.inference_audio, aud_mean, aud_std)
    
    # Model
    movement_net = MovementNet(args.d_input, args.d_output_body, args.d_output_rh, args.d_model, args.n_block, args.n_unet, args.n_attn, args.n_head, args.max_len, args.dropout,
                                   args.pre_lnorm, args.attn_type).to('cuda:0' if torch.cuda.is_available() else 'cpu')
    movement_net.load_state_dict(checkpoint['model_state_dict']['movement_net'])
    movement_net.eval()
    
    with torch.no_grad():
        print('inference...')
        X_test = torch.tensor(aud, dtype=torch.float32).to('cuda:0' if torch.cuda.is_available() else 'cpu').unsqueeze(0)
        lengths = X_test.size(1)
        lengths = torch.tensor(lengths).to('cuda:0' if torch.cuda.is_available() else 'cpu')
        lengths = lengths.unsqueeze(0)
        
        full_output = movement_net.forward(X_test, lengths)
            
        pred = full_output.squeeze(0)
        pred = pred.data.cpu().numpy()
    
        # Transform keypoints to world coordinate
        pred = pred * keypoints_std + keypoints_mean
        pred = np.reshape(pred, [len(pred), -1, 3])
      
    plot(args.inference_audio, args.plot_path, pred)
    with open(args.output_path, 'wb') as f:
        pickle.dump(pred, f)

if __name__ == '__main__':
    main()