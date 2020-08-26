import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn

from data import Download
from argument import parse
from metric import compute_pck, bowing_acc
from model.network import MovementNet
from visualize.animation import plot


def main():
    v_train = ['04'] # Training data only include No.4 violinist
    vid = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    p = 'flower' # test piece
    
    # Parser
    parser = parse()
    parser.add_argument('--plot_path', type=str, default='test.mp4', help='plot skeleton and add audio')
    parser.add_argument('--output_path', type=str, default='test.pkl', help='save skeletal data (only for no.9 violinist)')
    args = parser.parse_args()
    
    # Device
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = [i for i in range(len(args.gpu_ids.split(',')))]
    
    # Data
    download_data = Download()
    download_data.test_data()
    with open(download_data.test_dst, 'rb') as f:
        Data = pickle.load(f)
    keypoints_mean, keypoints_std = Data['keypoints_mean'], Data['keypoints_std']
    
    # Model
    checkpoint = torch.load(args.checkpoint)
    movement_net = MovementNet(args.d_input, args.d_output_body, args.d_output_rh, args.d_model, args.n_block, args.n_unet, args.n_attn, args.n_head, args.max_len, args.dropout,
                                   args.pre_lnorm, args.attn_type)
    movement_net = nn.DataParallel(movement_net, device_ids=gpu_ids)
    movement_net.load_state_dict(checkpoint['model_state_dict']['movement_net'])
    movement_net = movement_net.module
    movement_net.eval()
    
    #------------------------ START TESTING ----------------------------------#
    print('Testing... \n')
    result = {p: {}}
    l1 = []
    l1_hand = []
    pck_01 = []
    pck_02 = []
    bow = []
    bowx = []
    bowy = []
    bowz = []
    cosine = []
    with torch.no_grad():
        for v in vid:
            result[p][v] = {}
            aud = Data[p][v]['aud']
            keypoints = Data[p][v]['keypoints']
            sample_frame = Data[p][v]['sample_frame']
    
            X_test = torch.tensor(aud, dtype=torch.float32).cuda('cuda:' + str(gpu_ids[0]))
            lengths = X_test.size(1)
            lengths = torch.tensor(lengths).cuda('cuda:' + str(gpu_ids[0]))
            lengths = lengths.unsqueeze(0)
            
            full_output = movement_net.forward(X_test, lengths)
                
            pred = full_output.squeeze(0)
            pred = pred.data.cpu().numpy()
            targ = keypoints.squeeze(0)
            assert pred.shape==targ.shape
            
            # Transform keypoints to world coordinate
            pred = pred * keypoints_std + keypoints_mean
            pred = np.reshape(pred, [len(pred), -1, 3])
            targ = targ * keypoints_std + keypoints_mean
            targ = np.reshape(targ, [len(targ), -1, 3])
            
            # Clip time
            assert len(pred)==len(sample_frame)
            sample_time = sample_frame / 30
            sample_time = [sample_time[0], sample_time[-1]]
            
            result[p][v]['pred'] = pred
            result[p][v]['targ'] = targ
            result[p][v]['sample_time'] = sample_time
            
            # Evaluate test data of other 9 violinists
            if v not in v_train:
                v_l1 = np.mean(abs(pred - targ))
                v_l1_hand = np.mean(abs(pred[:, -1, :] - targ[:, -1, :]))
                v_pck_01 = compute_pck(pred, targ, alpha=0.1)
                v_pck_02 = compute_pck(pred, targ, alpha=0.2)
                v_bow_acc = bowing_acc(pred[:,-1,:], targ[:,-1,:], alpha=3) # only take right-hand wrist keypoints to calculate bowing attack accuracy
                v_cosine = np.mean(cosine_similarity(pred[:, -1, :], targ[:, -1, :]))
    
                l1.append(v_l1)
                l1_hand.append(v_l1_hand)
                pck_01.append(v_pck_01)
                pck_02.append(v_pck_02)
                bowx.append(v_bow_acc[0])
                bowy.append(v_bow_acc[1])
                bowz.append(v_bow_acc[2])
                bow.append(v_bow_acc[3])
                cosine.append(v_cosine)
        torch.cuda.empty_cache()
    
        avg_pck = (np.mean(pck_01) + np.mean(pck_02))*0.5
        print(p + ' Avg_L1_loss: %f' %np.mean(l1))
        print(p + ' Avg_L1_hand_loss: %f' %np.mean(l1_hand))
        print(p + ' Avg_Pck: %f' %avg_pck)
        print(p + ' Avg_Bowing_Attack_accuracyX: %f' %np.mean(bowx))
        print(p + ' Avg_Bowing_Attack_accuracyY: %f' %np.mean(bowy))
        print(p + ' Avg_Bowing_Attack_accuracyZ: %f' %np.mean(bowz))
        print(p + ' Avg_Bowing_Attack_accuracy: %f' %np.mean(bow))
        print(p + ' Avg_Cosine_Similarity: %f' %np.mean(cosine))
    
    if args.plot_path != None:    
        plot(download_data.wav_dst, args.plot_path, result[p]['09']['pred'], result[p]['09']['sample_time'])
    if args.output_path != None:
        with open(args.output_path, 'wb') as f:
            pickle.dump(result[p]['09'], f)

if __name__ == '__main__':
    main()