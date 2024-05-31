''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformer.Modules import ScaledDotProductAttention

__author__ = "Yu-Hsiang Huang"

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        """
        qkv这里在首次调用时是用了三个相同的输入
        """

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)  # 分别表示批次大小，查询序列的长度、键序列的长度、值序列的长度

        residual = q  # 保留了原始的输入，方便后续进行残差计算

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)  # 将原始的q经过线性变换进行重塑，作用是将其初始化成与输出相同维度大小的权重矩阵大小。这里是将所有的q按照多个头进行了拆分
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # 这里是将维度1和维度2进行交换位置
        # 这样做的目的是：多头自注意力中，通常需要将头的数量维度放在第二个位置，以便于并行计算不同头的注意力权重，从而提高计算效率。
        # 为什么不在为什么不可以直接使用q = self.w_qs(q).view(sz_b,  n_head,len_q, d_k)就进行交换的目的是：防止q、k、v混乱存储在同一个头中

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)  # 返回经过注意力计算后 的查询张量（经过了加权求和），以及注意力权重，表示每个查询对所有键的注意力分配情况

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)  # 重组多头注意力，将注意力头和查询长度的维度进行交换，\
        # 然后使用contiguious消除转置可能会带来的张量存储不连续问题，然后再将长度后的所有维度数据合并成一个维度.这包含所有的查询信息，并且每个查询位置的输出都将包含来自所有头的信息
        q = self.dropout(self.fc(q))  # 将最终的查询经过全连接层输出成d_model的维度
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x
