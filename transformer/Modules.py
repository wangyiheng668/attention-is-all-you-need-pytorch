import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Yu-Hsiang Huang"

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))  # 原来的k[batch_size,n_head, len_k, d_k],\
        # 交换以后是[batch_size,n_head, d_k, len_k],这里的目的是方便将其与q进行矩阵乘法（点积）


        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)  # 这里会将掩码为0的地方设置为一个很大的负数，这样经过softmax后就变成了0

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)  # 得到注意力权重以后将其与v进行相乘

        return output, attn
