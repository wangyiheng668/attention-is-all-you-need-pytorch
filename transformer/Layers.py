''' Define the Layers '''
import torch.nn as nn
import torch
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward


__author__ = "Yu-Hsiang Huang"


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)  #
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    '''
    Compose with three layers
    假设我们在做机器翻译，将英文翻译成中文，输入句子是 "I love programming"，目标句子是 "我 爱 编程"。
    第一层：自身的注意力，解码器的输入是目标句子的部分，如 <SOS>、我、爱 等。解码器通过自注意力机制来处理目标句子内部的依赖关系。
    第二层：与编码的注意力，解码器的输入序列已经处理过，编码器的输出 [enc_out_1, enc_out_2, enc_out_3] 是源语言 "I love programming" 的表示。/
            编码-解码注意力机制会根据解码器的当前状态（比如正在生成 "我"）来计算应关注源句子的哪个部分。例如，解码器可能会发现 "I" 对应 "我"，并给 "I" 一个较大的权重，表示需要更多关注 "I" 的信息。
    '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)  # 解码器自身的序列注意力
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)  # 解码器结合编码器中输入序列的注意力
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):  # 这里需要将上一层的解码器输出作为当前层的解码器输入，也就是这里的dec_input
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)  # 传入的分别是Q；K；V，也就是探寻解码器输入序列之间的注意力，也就是探寻除了输入序列之外之前的解码器输出与当前的关系
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)  # 探寻当前层解码器输出和编码器输出（输入序列）的注意力关系，其中的mask是输入序列和最大长度序列的相应mask，也就是说我不能看到输入序列之后的东西，这里只是探寻当前的解码器输出和输入序列的相对关系
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn
