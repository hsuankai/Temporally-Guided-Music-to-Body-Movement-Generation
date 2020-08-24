import math
import string
import numpy as np

import torch
import torch.nn as nn
from .positional_embedding import DistanceEmbedding, PositionEmbeddingType, KeyStartPosition, EmbeddingPaddingMode


class GlobalMultiheadAttention(nn.Module):
     """
     Global Multihead Attention
     """
     def __init__(self, d_k, dropout):
         super(GlobalMultiheadAttention, self).__init__()
         self.d_k = d_k
         self.attn_dropout = nn.Dropout(p=dropout)
         
     def forward(self, query, key, value, mask=None, query_mask=None):
         bz, h, seq_q, d_q = query.size()
         bz, h, seq_k, d_k = key.size()
         
         query = query.view(-1, seq_q, d_q) 
         key = key.view(-1, seq_k, d_k)
         
         # Get attention score
         attn = torch.bmm(query, key.transpose(1, 2))
         attn = attn / math.sqrt(self.d_k)

         # Masking to ignore padding (key side)
         if mask is not None:
             attn = attn.masked_fill(mask, -2 ** 32 + 1)
             attn = torch.softmax(attn, dim=-1)
         else:
             attn = torch.softmax(attn, dim=-1)

         # Masking to ignore padding (query side)
         if query_mask is not None:
             attn = attn * query_mask

         # Dropout
         attn = self.attn_dropout(attn)

         # Get Context Vector
         output = torch.bmm(attn, value)
         return output, attn

class RelMultiheadAttention(nn.Module):
     """
     Relative Multihead Attention
     This code is a modified version from https://github.com/Separius/CudaRelativeAttention/blob/master/relative_attention.py
     """
     def __init__(self, d_k, n_head, max_len, dropout):
         super(RelMultiheadAttention, self).__init__()
         self.rel_attn = RelativeAttention1d(n_head, d_k * n_head, max_len, heads_share_relative_embeddings=True, 
                                             embedding_padding_modes=EmbeddingPaddingMode.Extend,
                                             position_embedding_types=PositionEmbeddingType.Hybrid,
                                             key_start_positions=KeyStartPosition.BeforeQuery, 
                                             add_bias_to_query_for_relative_logits=True,
                                             add_bias_to_query_for_key_logit=True)
         self.d_k = d_k
         self.attn_dropout = nn.Dropout(p=dropout)
         
     def forward(self, query, key, value, mask=None, query_mask=None):   
         # Get attention score
         attn = self.rel_attn(query, key)
         attn = attn / math.sqrt(self.d_k)

         # Masking to ignore padding (key side)
         if mask is not None:
             attn = attn.masked_fill(mask, -2 ** 32 + 1)
             attn = torch.softmax(attn, dim=-1)
         else:
             attn = torch.softmax(attn, dim=-1)

         # Masking to ignore padding (query side)
         if query_mask is not None:
             attn = attn * query_mask

         # Dropout
         attn = self.attn_dropout(attn)

         # Get Context Vector
         output = torch.bmm(attn, value)
         return output, attn




class RelativeAttention(nn.Module):
    def __init__(self, n_dim, num_heads, model_depth, max_relative_positions_past,
                 max_relative_positions_future=None, heads_share_relative_embeddings=True,
                 embedding_padding_modes=EmbeddingPaddingMode.Extend,
                 position_embedding_types=PositionEmbeddingType.Hybrid,
                 key_start_positions=KeyStartPosition.BeforeQuery,
                 add_bias_to_query_for_relative_logits=True,  # the d term in transformer-xl(second equation in page 5)
                 add_bias_to_query_for_key_logit=True,  # the c term in transformer-xl(second equation in page 5)
                 use_custom_cuda_kernel=True):
        super(RelativeAttention).__init__()
        assert model_depth % num_heads == 0
        assert 1 <= n_dim <= 3
        self.use_custom_cuda_kernel = use_custom_cuda_kernel
        self.head_depth = model_depth // num_heads
        self.n_dimension = n_dim
        self.num_heads = num_heads
        max_relative_positions_past = self._get_list(max_relative_positions_past, int)
        if max_relative_positions_future is None:
            max_relative_positions_future = max_relative_positions_past
        else:
            max_relative_positions_future = self._get_list(max_relative_positions_future, int)
        heads_share_relative_embeddings = self._get_list(heads_share_relative_embeddings, bool)
        embedding_padding_modes = self._get_list(embedding_padding_modes, EmbeddingPaddingMode)
        position_embedding_types = self._get_list(position_embedding_types, PositionEmbeddingType)
        key_start_positions = self._get_list(key_start_positions, KeyStartPosition)
        add_bias_to_query_for_relative_logits = self._get_list(add_bias_to_query_for_relative_logits, bool)
        self.relative_biases = []
        for i in range(n_dim):
            new_param = nn.Parameter(torch.randn(self.head_depth, num_heads) * 0.01) \
                if add_bias_to_query_for_relative_logits[i] else None
            self.register_parameter('relative_bias_{}'.format(i + 1), new_param)
            self.relative_biases.append(new_param)
        if add_bias_to_query_for_key_logit:
            self.query_to_key_bias = nn.Parameter(torch.randn(num_heads, self.head_depth) * 0.01)
        else:
            self.register_parameter('query_to_key_bias', None)
        self.relative_embeddings = nn.ModuleList([DistanceEmbedding(self.head_depth, max_relative_positions_past[i],
                                                                    max_relative_positions_future[i], num_heads,
                                                                    heads_share_relative_embeddings[i],
                                                                    embedding_padding_modes[i],
                                                                    position_embedding_types[i],
                                                                    key_start_positions[i]) for i in range(n_dim)])

    def _get_list(self, optional_list, desired_class):
        if not isinstance(optional_list, (list, tuple)):
            obj_list = [optional_list] * self.n_dimension  # w, h, t
        else:
            obj_list = optional_list
        desired_list = []
        for obj in obj_list:
            if desired_class == int:
                if isinstance(obj, int):
                    desired_list.append(obj)
                else:
                    desired_list.append(int(obj))
            elif desired_class == bool:
                if isinstance(obj, bool):
                    desired_list.append(obj)
                else:
                    desired_list.append(bool(obj))
            else:  # enum cases
                if isinstance(obj, desired_class):
                    desired_list.append(obj)
                elif isinstance(obj, str):
                    desired_list.append(desired_class[obj])
                elif isinstance(obj, int):
                    desired_list.append(desired_class(obj))
                else:
                    raise ValueError(f'invalid input({obj}) for enum {desired_class}')
        return desired_list

    @staticmethod
    def relative_position_to_absolute_position(x):
        """Converts tensor from relative to aboslute indexing for local attention.
        Args:
            x: [batch (or batch*num_blocks), heads, length, 2 * length - 1]
        Returns:
            A Tensor of shape [batch (or batch*num_blocks), heads, length, length]
        """
        batch, heads, length, _ = x.size()
        # Concat columns of pad to shift from relative to absolute indexing.
        col_pad = torch.zeros((batch, heads, length, 1), device=x.device, dtype=x.dtype)
        x = torch.cat([x, col_pad], dim=3)
        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        flat_x = x.reshape(batch, heads, length * 2 * length)
        flat_pad = torch.zeros((batch, heads, length - 1), device=x.device, dtype=x.dtype)
        flat_x_padded = torch.cat([flat_x, flat_pad], dim=2)
        # Reshape and slice out the padded elements.
        final_x = flat_x_padded.reshape(batch, heads, length + 1, 2 * length - 1)
        return final_x[:, :, :length, length - 1:]

    def forward(self, q, k, mask=None):
        raise NotImplementedError()

    @staticmethod
    def apply_mask(logits, mask):
        print(logits.size())
        if mask is not None:
            if mask.ndimension() == 2:
                mask = mask.unsqueeze(0)
            return logits + mask.to(logits.dtype) * -10000.0
#            return logits.masked_fill_(mask, -2 ** 32 + 1)
        return logits

    def get_logits(self, q, k):
        # q is (B, N, ..., d) and k is also (B, N, ..., d); Note that m,n,o are in the middle of alphabet
        # => logits with shape == (B * N, Sq, Sk)
        if self.query_to_key_bias is not None:
            # q is (B, N, ..., d) and bias is (N, d)
            q = q + self.query_to_key_bias.view(1, q.size(1), *([1] * (q.ndimension() - 3)), -1)
        return torch.einsum(
            'mn{q_dims}o, mn{k_dims}o -> mn{q_dims}{k_dims}'.format(q_dims=string.ascii_lowercase[:q.ndimension() - 3],
                                                                    k_dims=string.ascii_lowercase[::-1][:k.ndimension() - 3]),
            q, k).view(q.size(0) * q.size(1), np.prod(q.size()[2:-1]), np.prod(k.size()[2:-1]))


class RelativeAttention1d(RelativeAttention):
    def __init__(self, num_heads, model_depth, max_relative_positions_past, max_relative_positions_future=None,
                 heads_share_relative_embeddings=True, embedding_padding_modes=EmbeddingPaddingMode.Extend,
                 position_embedding_types=PositionEmbeddingType.Hybrid,
                 key_start_positions=KeyStartPosition.BeforeQuery, add_bias_to_query_for_relative_logits=True,
                 add_bias_to_query_for_key_logit=True):
        super().__init__(1, num_heads, model_depth, max_relative_positions_past, max_relative_positions_future,
                         heads_share_relative_embeddings, embedding_padding_modes, position_embedding_types,
                         key_start_positions, add_bias_to_query_for_relative_logits, add_bias_to_query_for_key_logit,
                         use_custom_cuda_kernel=False)

    def forward(self, q, k, mask=None):
        """forward function for RelativeAttention.
            Args:
                q: [batch, heads, Wq, d]
                k: [batch, heads, Wk, d]
                mask: Optional[binary tensor of shape [batch * heads or None, Wq, Wk]]
                        true to mask(add -10000.0) and false to attend
            Returns:
                logits: [batch * heads, Wq, Wk]
        """
        if self.use_custom_cuda_kernel:
            raise ValueError('can not use custom cuda kernel with 1d')
        if not q.size() == k.size():
            raise ValueError('RelativeAttention1d only supports self attention so q.size() == k.size()')
        batch, num_heads, width, _ = q.size()
        logits = self.get_logits(q, k)
        distance_logits = self.relative_embeddings[0](width, q, self.relative_biases[0])
        width_rel_logits = self.relative_position_to_absolute_position(distance_logits).view_as(logits)
        attn_score = logits + width_rel_logits
        return attn_score
