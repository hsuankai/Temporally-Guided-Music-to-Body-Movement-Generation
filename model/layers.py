import torch
import torch.nn as nn

from .module import Down, Up
from .positional_embedding import PositionalEncoding
from .attention import Self_attention, FFN_linear
from .utils import lengths_to_tensor

    
class AttentionNetwork(nn.Module):
    """
    Global/Relative attention network with positionwise feed-forward network
    """
    def __init__(self, d_model, n_attn, n_head, max_len, dropout, pre_lnorm=False, attn_type='rel'):
        super(AttentionNetwork, self).__init__()
        self.attn_type = attn_type
        
        # Absolute positional embedding
        if attn_type=='abs':
            self.pos_emb = PositionalEncoding(d_model, max_len+1)
            self.alpha = nn.Parameter(torch.ones(1))
            self.pos_dropout = nn.Dropout(0.1)
            
        self.slf_attn = [Self_attention(d_model, n_head, max_len, dropout, pre_lnorm, attn_type)] * n_attn
        self.slf_attn = nn.Sequential(*self.slf_attn)
        
        # Postionwise network
        self.pos_ffn = [FFN_linear(d_model, dropout, pre_lnorm)] * n_attn
        self.pos_ffn = nn.Sequential(*self.pos_ffn)
        
    def forward(self, x, x_pos, return_attns=False):    
        c_mask = x_pos.ne(0).type(torch.float)
        mask = x_pos.eq(0).unsqueeze(1).repeat(1, x.size(1), 1)
        
        if self.attn_type=='abs':
            # Get positional embedding, apply alpha and add
            pos = self.pos_emb(x_pos)
            x = pos * self.alpha + x
            x = self.pos_dropout(x)
        
        attns = []
        for layer, ffn in zip(self.slf_attn, self.pos_ffn):
            x, attn = layer(x, x, mask=mask, query_mask=c_mask)
            if return_attns:    
                attns.append(attn)
            x *= c_mask.unsqueeze(-1)
            x = ffn(x)
            x *= c_mask.unsqueeze(-1)
            
        if return_attns:
            return x, attns
        else:
            return x


class Unet_block(nn.Module):
    """
    U-net block including attention network
    """
    def __init__(self, d_model, n_unet, n_attn, n_head, max_len, dropout, pre_lnorm, attn_type):
        super(Unet_block, self).__init__()
        """
        Args:
             d_model: dimension of input
             n_head: number of head
             max_len: the max length of input sequence
             dropout: dropout rate
             pre_lnorm: use pre-layer normalization or not
             attn_type: use global attention or relative attention
            
        """             
        self.max_len = max_len
        self.n_attn = n_attn
        
        self.down = nn.ModuleList([Down(d_model, d_model, residual=False)] * n_unet)
        self.up = nn.ModuleList([Up(d_model*2, d_model, residual=False)] * n_unet)    
        self.attn_net = AttentionNetwork(d_model, n_attn, n_head, max_len, dropout, pre_lnorm, attn_type)
        
    def forward(self, x, lengths, return_attns):
        # Down stream
        x = x.transpose(1,2)
        skip = [x]
        for i, down in enumerate(self.down):
            x = down(x)
            if i != len(self.down) - 1:
                skip.append(x)
        
        # Self-attention
        if self.n_attn > 1:
            x = x.transpose(1,2)
            x_pos = lengths_to_tensor(lengths, max_len=self.max_len)[:, :x.size(1)] # lengths to matrix
            if return_attns:
                x, attns = self.attn_net(x, x_pos, return_attns=return_attns)
            else:
                x = self.attn_net(x, x_pos, return_attns=return_attns)
            x = x.transpose(1,2)
        
        # Up stream
        for i, up in enumerate(self.up):
            x = up(x, skip[-(i+1)])
            
        x = x.transpose(1,2)
        return x