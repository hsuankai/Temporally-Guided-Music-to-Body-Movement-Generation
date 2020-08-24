import torch
import torch.nn as nn

from .module import Linear
from .multihead_attention import GlobalMultiheadAttention, RelMultiheadAttention


class Self_attention(nn.Module):
     """
     This is the implementation of self-attention mechanism proposed in Attention Is All You Need [https://arxiv.org/abs/1706.03762], and
     we add implementation of relative attention proposed in Music Transformer [https://magenta.tensorflow.org/music-transformer]
     """
     def __init__(self, d_model, n_head, max_len, dropout, pre_lnorm=False, attn_type='rel'):
         super(Self_attention, self).__init__()
         self.d_model = d_model
         self.d_q = d_model // n_head
         self.d_k = d_model // n_head
         self.d_v = d_model // n_head
         self.n_head = n_head
         self.pre_lnorm = pre_lnorm

         self.query = Linear(d_model, self.d_q * n_head, bias=False)
         self.key = Linear(d_model, self.d_k * n_head, bias=False)
         self.value = Linear(d_model, self.d_v * n_head, bias=False)

         if attn_type=='abs':
             self.multihead = GlobalMultiheadAttention(self.d_k, dropout)
         elif attn_type=='rel':
             self.multihead = RelMultiheadAttention(self.d_k, n_head, max_len, dropout)

         self.residual_dropout = nn.Dropout(p=dropout)

         self.final_linear = Linear(d_model*2, d_model, bias=False)

         self.layer_norm = nn.LayerNorm(d_model)

     def forward(self, memory, decoder_input, mask=None, query_mask=None):
         batch_size = memory.size(0)
         seq_k = memory.size(1)
         seq_q = decoder_input.size(1)

         residual = decoder_input

         if self.pre_lnorm:
             memory = self.layer_norm(memory)
             decoder_input = self.layer_norm(decoder_input)

         # Repeat masks h times
         if query_mask is not None:
             query_mask = query_mask.unsqueeze(-1).repeat(1, 1, seq_k)
             query_mask = query_mask.repeat(self.n_head, 1, 1)
         if mask is not None:
             mask = mask.repeat(self.n_head, 1, 1)

         # Make multihead
         query = self.query(decoder_input).view(batch_size, seq_q, self.n_head, self.d_q)
         key = self.key(memory).view(batch_size, seq_k, self.n_head, self.d_k)
         value = self.value(memory).view(batch_size, seq_k, self.n_head, self.d_v)

         query = query.permute(0, 2, 1, 3).contiguous() # bz x h x qlen x dq
         key = key.permute(0, 2, 1, 3).contiguous() # bz x h, klen x dk
         value = value.permute(0, 2, 1, 3).contiguous().view(-1, seq_k, self.d_v) # h x bz, qlen, dv

         # Get context vector
         output, attns = self.multihead(query, key, value, mask=mask, query_mask=query_mask)

         # Concatenate all multihead context vector
         output = output.view(self.n_head, batch_size, seq_q, self.d_v)
         output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_q, -1)

         # Concatenate context vector with input (most important)
         output = torch.cat([decoder_input, output], dim=-1) #

         # Final linear
         output = self.final_linear(output)

         # Residual dropout
         output = self.residual_dropout(output)

         # Residual dropout & connection
         output = output + residual

         # Layer normalization
         if not self.pre_lnorm:
             output = self.layer_norm(output)
         return output, attns


class FFN_linear(nn.Module):
    """
    Positionwise Feed-Forward Network
    """
    def __init__(self, d_model, dropout, pre_lnorm=False, layer_norm_epsilon=1e-5):
        super(FFN_linear, self).__init__()
        self.FFN = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(d_model*4, d_model),
            nn.Dropout(p=dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.pre_lnorm = pre_lnorm

    def forward(self, x):
        residual = x
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            output = self.FFN(self.layer_norm(x))

            # residual connection
            output = output + residual
        else:
            # positionwise feed-forward
            output = self.FFN(x)

            # residual connection + layer normalization
            output = self.layer_norm(output + residual)
        return output
