__all__ = ['PatchTST_backbone']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from torch.nn.init import trunc_normal_
#from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.RevIN import RevIN
import math
# Cell
class PatchTST_backbone(nn.Module):
    def __init__(self,configs, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                 verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1) # patch的数量是(336-16)/8 + 1 向下取整
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) # 使用了 PyTorch 中的 nn.ReplicationPad1d 模块，用于对一维张量进行复制填充操作
            patch_num += 1
        print(patch_num)
        # Backbone 
        self.backbone = TSTiEncoder(configs,c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.head_nf = d_model * patch_num 
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten': 
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
        
    
    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
        # norm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)
            
        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]
        
        # model
        z = self.backbone(z)                                                                # z: [bs x nvars x d_model x patch_num]
        z = self.head(z)                                                                    # z: [bs x nvars x target_window] 
        
        # denorm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        return z
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x
        
        
    
    
class TSTiEncoder(nn.Module):  #i means channel-independent
    def __init__(self,configs, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):
        
        
        super().__init__()
        
        self.patch_num = patch_num
        self.patch_len = patch_len
        
        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len # patch数量看作时序长度

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(configs,q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)

        
    def forward(self, x) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]
        
        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0,1,3,2)                                                   # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)                                                          # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))      # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x patch_num x d_model]

        # Encoder
        z = self.encoder(u)                                                      # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))                # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]
        
        return z    
            
            
    
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, configs, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()
        # 0层和2层是正常window，1层是shift_window
        window_size = configs.window_size
        shift_size = math.ceil(window_size/2)
        PM = False # Patch Merging
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            #window_size = window_size/()
            self.layers.add(TSTEncoderLayer(configs,
                                                    window_size
                                                     ,
                                                     0 if (i % 2 == 0) else shift_size
                                                     , 
                                                      q_len
                                                     , 
                                                     d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn))
        self.res_attention = res_attention

    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output

class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(2 * dim, 2 * dim, bias=False)
        
        #self.norm = norm_layer(2 * dim)
        self.norm = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(2*dim), Transpose(1,2))
        
    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, L, C).
        """
        B, L, C = x.shape

        # padding
        pad_input = (L % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, L % 2))

        x0 = x[:, 0::2, :]  # B L/2 C
        x1 = x[:, 1::2, :]  # B L/2 C
        x = torch.cat([x0, x1], -1)  # B L/2 2*C

        x = self.norm(x)
        x = self.reduction(x)

        return x
def window_partition(x, window_size):
    """
    Args:
        x: (B, L, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, C)
    """
    B, L, C = x.shape
    x = x.view(B, L // window_size, window_size, C)
    windows = x.permute(0, 1, 2, 3).contiguous().view(-1, window_size, C)
    return windows

def compute_mask(L, window_size, shift_size):
    Lp = int(np.ceil(L / window_size)) * window_size
    img_mask = torch.zeros((1, Lp, 1))  # 1 Lp 1
    pad_size = int(Lp - L)
    if (pad_size == 0) or (pad_size + shift_size == window_size):
        segs = (slice(-window_size), slice(-window_size, -
                shift_size), slice(-shift_size, None))
    elif pad_size + shift_size > window_size:
        seg1 = int(window_size * 2 - L + shift_size)
        segs = (slice(-seg1), slice(-seg1, -window_size),
                slice(-window_size, -shift_size), slice(-shift_size, None))
    elif pad_size + shift_size < window_size:
        seg1 = int(window_size * 2 - L + shift_size)
        segs = (slice(-window_size), slice(-window_size, -seg1),
                slice(-seg1, -shift_size), slice(-shift_size, None))
    cnt = 0
    for d in segs:
        img_mask[:, d, :] = cnt
        cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws, 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(
        attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask
class TSTEncoderLayer(nn.Module):
    def __init__(self, configs, window_size,shift_size, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.shift_size = shift_size
        if self.shift_size!=0:
            shift_size = int(window_size//2)
        if self.shift_size>0:
            self.mask = compute_mask(q_len, window_size, shift_size)
        # Multi-Head attention
        self.res_attention = res_attention
        # self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        self.self_attn = WindowAttention1D(configs,window_size,shift_size,dim=d_model, num_heads=n_heads, qkv_bias=True, qk_scale=None, attn_drop=attn_dropout, proj_drop=dropout)
        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),#nn.GELU()
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        # if self.res_attention:
        #     src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # else:
        #     src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # if self.store_attn:
        #     self.attn = attn
        if self.shift_size>0:
            src2 = self.self_attn(src,self.mask.to(src.device))
        else:
            src2 = self.self_attn(src)
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        # if self.res_attention:
        #     return src, scores
        # else:
        #     return src
        return src



# class _MultiheadAttention(nn.Module):
#     def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
#         """Multi Head Attention Layer
#         Input shape:
#             Q:       [batch_size (bs) x max_q_len x d_model]
#             K, V:    [batch_size (bs) x q_len x d_model]
#             mask:    [q_len x q_len]
#         """
#         super().__init__()
#         d_k = d_model // n_heads if d_k is None else d_k
#         d_v = d_model // n_heads if d_v is None else d_v

#         self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

#         self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
#         self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
#         self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

#         # Scaled Dot-Product Attention (multiple heads)
#         self.res_attention = res_attention
#         self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

#         # Poject output
#         self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


#     def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
#                 key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

#         bs = Q.size(0)
#         if K is None: K = Q
#         if V is None: V = Q

#         # Linear (+ split in multiple heads)
#         q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
#         k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
#         v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

#         # Apply Scaled Dot-Product Attention (multiple heads)
#         if self.res_attention:
#             output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
#         else:
#             output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
#         # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

#         # back to the original inputs dimensions
#         output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
#         output = self.to_out(output)

#         if self.res_attention: return output, attn_weights, attn_scores
#         else: return output, attn_weights


# class _ScaledDotProductAttention(nn.Module):
#     r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
#     (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
#     by Lee et al, 2021)"""

#     def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
#         super().__init__()
#         self.attn_dropout = nn.Dropout(attn_dropout)
#         self.res_attention = res_attention
#         head_dim = d_model // n_heads
#         self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
#         self.lsa = lsa

#     def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
#         '''
#         Input shape:
#             q               : [bs x n_heads x max_q_len x d_k]
#             k               : [bs x n_heads x d_k x seq_len]
#             v               : [bs x n_heads x seq_len x d_v]
#             prev            : [bs x n_heads x q_len x seq_len]
#             key_padding_mask: [bs x seq_len]
#             attn_mask       : [1 x seq_len x seq_len]
#         Output shape:
#             output:  [bs x n_heads x q_len x d_v]
#             attn   : [bs x n_heads x q_len x seq_len]
#             scores : [bs x n_heads x q_len x seq_len]
#         '''

#         # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
#         attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

#         # Add pre-softmax attention scores from the previous layer (optional)
#         if prev is not None: attn_scores = attn_scores + prev

#         # Attention mask (optional)
#         if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
#             if attn_mask.dtype == torch.bool:
#                 attn_scores.masked_fill_(attn_mask, -np.inf)
#             else:
#                 attn_scores += attn_mask

#         # Key padding mask (optional)
#         if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
#             attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

#         # normalize the attention weights
#         attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
#         attn_weights = self.attn_dropout(attn_weights)

#         # compute the new values given the attention weights
#         output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

#         if self.res_attention: return output, attn_weights, attn_scores
#         else: return output, attn_weights

# 输入 B L C 输出 B L C
class WindowAttention1D(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (int): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, configs,window_size,shift_size, dim,  num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.shift_size = shift_size
        self.dim = dim
        self.window_size = window_size  # Wl
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size - 1), num_heads))  # 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_l = torch.arange(self.window_size)
        # coords = torch.stack(torch.meshgrid(
        #     [coords_l], indexing='ij'))  # 1, Wl
        coords = torch.stack(torch.meshgrid(
            [coords_l]))  # 1, Wl
        # 使用 torch.unsqueeze 和 torch.repeat 创建坐标网格
        # coords = coords_l.unsqueeze(0).repeat(1, window_size)
        print(coords)
        
        coords_flatten = torch.flatten(coords, 1)  # 1, Wl
        relative_coords = coords_flatten[:, :, None] - \
            coords_flatten[:, None, :]  # 1, Wl, Wl
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wl, Wl, 2
        relative_coords[:, :, 0] += self.window_size - \
            1  # shift to start from 0
        # relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wl, Wl
        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
    def window_partition(self,x):
            """
            Args:
                x: (B, L, C)
                window_size (int): window size

            Returns:
                windows: (num_windows*B, window_size, C)
            """
            #在这之前需要填充
            B, L, C = x.shape
            x = x.view(B, L // self.window_size, self.window_size, C)
            windows = x.permute(0, 1, 2, 3).contiguous().view(-1, self.window_size, C)
            return windows

    def window_reverse(self,windows, window_size, L):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size (int): Window size
            L (int): Sequence length

        Returns:
            x: (B, L, C)
        """
        B = int(windows.shape[0] / (L / window_size))
        x = windows.view(B, L // window_size, window_size, -1)
        x = x.permute(0, 1, 2, 3).contiguous().view(B, L, -1)
        return x
    def pad_shift_x(self,x,shift_size):
        B, L, C = x.shape
        #shift_size = 0
        pad_l = 0
        pad_r = (self.window_size - L % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r))
        _, Lp, _ = x.shape  
        if shift_size > 0:
            shifted_x = torch.roll(x, shifts=-shift_size, dims=(1))
            #attn_mask = None#mask_matrix，前面计算过了，直接传过来
        else:
            shifted_x = x
            #attn_mask = None      
        x_windows = self.window_partition(shifted_x)
        return x_windows,Lp      
        
    def forward(self, x, mask=None):
        # pad feature maps to multiples of window size
        B, L, C = x.shape
        shift_size = self.shift_size
        x,Lp = self.pad_shift_x(x,shift_size) # B L C -> B Lp C
        x = self.window_partition(x) # B Lp C -> B windows_num windows_size C->B*wn windows C
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wl, Wl) or None
        """
        B_, N, C = x.shape
        #print(x.shape) # torch.Size([6272, 6, 128])
        qkv = self.qkv(x)
        #print(qkv.shape) torch.Size([6272, 6, 384])
        qkv=qkv.reshape(B_, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
       # print(q.shape)  # torch.Size([6272, 4, 6, 32])

        attn = (q @ k.transpose(-2, -1))
       # print(attn.shape) #torch.Size([6272, 4, 6, 6])
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size, self.window_size, -1)  # Wl,Wl,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wl, Wl
        
        #print(attn.shape,relative_position_bias.shape) #torch.Size([6272, 4, 6, 6]) torch.Size([144, 6, 6])
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(-1, *(self.window_size, C))
        x = self.window_reverse(x, self.window_size, Lp)
        if shift_size > 0:
            x = torch.roll(x, shifts=shift_size, dims=(1))
        else:
            x = x
        x = x[:, :L, :].contiguous() # B L C

        return x