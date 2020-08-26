import argparse


def parse():
    parser = argparse.ArgumentParser(description='Body Movement Network')
    
    # Global arguments
    parser.add_argument('--epoch', type=int, default=200, help='epoch')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--early_stop_iter', type=int, default=10, help='use early stopping scheme if > 0')
    parser.add_argument('--checkpoint', type=str, default='checkpoint/best.pth', help='the path of checkpoint')
    parser.add_argument('--gpu_ids', type=str, default='0', help='specify gpu ids, you can also use multi-gpu, e.g. 0,1,2') 
    
    # Model arguments
    parser.add_argument('--d_input', type=int, default=28, help='the hidden units used in network')
    parser.add_argument('--d_model', type=int, default=512, help='the hidden units used in network')
    parser.add_argument('--d_output_body', type=int, default=39, help='the number of body points')
    parser.add_argument('--d_output_rh', type=int, default=6, help='the number of right-hand points')
    parser.add_argument('--warmup_steps', type=int, default=500, help='use warm-up scheme if > 0')
    parser.add_argument('--n_block', type=int, default=2, help='the number of U-net block')
    parser.add_argument('--n_unet', type=int, default=4, help='the number of U-net layer')
    parser.add_argument('--n_attn', type=int, default=1, help='the number of self-attention layer')
    parser.add_argument('--n_head', type=int, default=4, help='the number of head')
    parser.add_argument('--max_len', type=int, default=900, help='the max length of sequence')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--pre_lnorm', type=bool, default=False, help='applying pre-layer normalization or not')
    parser.add_argument('--attn_type', type=str, default='rel', help='the type of self-attention') 
    return parser