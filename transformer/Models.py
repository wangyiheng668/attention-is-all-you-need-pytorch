''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from transformer.Layers import EncoderLayer, DecoderLayer


__author__ = "Yu-Hsiang Huang"


def get_pad_mask(seq, pad_idx):
    """
    这个函数的作用是为了屏蔽当前索引的编码位置，即不关注当前的位置
    """
    return (seq != pad_idx).unsqueeze(-2)  # 这里在倒数第二维上面添加一个维度是为了匹配其它数据的需要


def get_subsequent_mask(seq):
    """
    For masking out the subsequent info
    主要作用是为了屏蔽此位置之后的所有信息
    这里的主要作用是根据输入序列的大小创建一个(1, len_s, len_s) 的张量，并将其上三角矩阵的元素（包括对角线元素）设置维1
    - return: 一个主对角以下元素（包括主对角元素）为Ture，主对角以上元素为false的矩阵
    """
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):
    """

    """

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)  # 使用 .clone().detach() 是为了避免在计算图中跟踪位置表的任何修改，保证位置表作为常量存在。

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()  # pos_table就是计算位置编码的三角公式


class Encoder(nn.Module):
    """
    - A encoder model with self attention mechanism.
    n_src_vocab：源语言词汇表的大小，即源语言的词汇数量。
    d_word_vec：词向量的维度大小，表示每个单词在词向量空间中的表示维度。
    n_layers：编码器（encoder）或解码器（decoder）的层数，决定了模型的深度。
    n_head：注意力头的数量，表示每个多头自注意力机制中的注意力头的个数。
    d_k 和 d_v：每个注意力头中 key 和 value 的维度大小。
    d_model：模型的总体维度大小，也就是每个位置输入向量的维度。
    d_inner：全连接层中间隐藏层的维度大小。
    pad_idx：用于填充的索引值，在处理变长序列时会用到。
    dropout：Dropout 概率，用于防止过拟合。
    n_position：位置编码的最大序列长度，用于表示序列中不同位置的信息。
    scale_emb：一个布尔值，表示是否对词嵌入进行缩放。
    """

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False):

        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)  # 词向量嵌入
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)  # 根究词响亮的维度大小和位置编码的最大长度进行位置编码
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])  # 这里构建了一个层堆栈，每一层都有一个相同的结构，即有多个encoderlayer，并且可以使这些模块可以像列表一样进行索引，并且每一个层都使用了相同的参数进行初始化
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)  # 这里是初始化层归一化，
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):
        """
        src_seq：输入的源序列，通常是一个整数张量，表示词的索引。
        src_mask：用于屏蔽（masking）源序列中无效（如填充或非目标）数据的掩码。
        return_attns：一个布尔值，决定是否返回每个编码层中的自注意力权重。
        """

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.src_word_emb(src_seq)  # 嵌入层输出
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5  #这里就是论文中提到的维度缩放的概念，详见论文笔记
        enc_output = self.dropout(self.position_enc(enc_output))  # 然后对嵌入层进行位置编码
        enc_output = self.layer_norm(enc_output)  # 对于输出进行层归一化

        for enc_layer in self.layer_stack:  # 对于encodelayer堆栈中的每一个层
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)  # 这里计算了编码后的输出（先嵌入向量然后进行位置编码再送到编码器中），返回的是编码后的输出，以及计算的注意力得分
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:  # 这里代表是否返回每个编码层中的自注意力权重
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1, scale_emb=False):

        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class Transformer(nn.Module):
    """
    A sequence to sequence model with attention mechanism.
    n_src_vocab: 源语言词汇表的大小（vocab size）。

    n_trg_vocab: 目标语言词汇表的大小（vocab size）。

    src_pad_idx: 源语言中用于填充的索引。在序列中的 padding 部分通常会用一个特殊的 token（例如 <pad>）进行填充，这个参数指定了该 token 在源语言词汇表中的索引。

    trg_pad_idx: 目标语言中用于填充的索引。与 src_pad_idx 类似，这个参数指定了目标语言中用于填充的 token 在目标语言词汇表中的索引。

    d_word_vec: 词向量的维度（dimension of word vectors）。默认值为 512。

    d_model: 模型的隐藏层维度（dimension of the model）。默认值为 512。

    d_inner: 在多头自注意力和前馈神经网络中，内部层（inner layers）的维度。默认值为 2048。

    n_layers: 模型中的层数（number of layers）。默认值为 6。

    n_head: 多头自注意力机制中的注意力头数（number of attention heads）。默认值为 8。

    d_k: 注意力头中键（key）的维度（dimension of keys）。默认值为 64。

    d_v: 注意力头中值（value）的维度（dimension of values）。默认值为 64。

    dropout: 在模型中使用的 dropout 概率。默认值为 0.1。

    n_position: 序列中位置编码的最大长度（maximum length of positional encoding）。默认值为 200。

    trg_emb_prj_weight_sharing: 是否共享目标语言嵌入层和投影层的权重。默认值为 True。针对此请详见wyh针对这篇文章的笔记

    emb_src_trg_weight_sharing: 是否共享源语言和目标语言嵌入层的权重。默认值为 True。

    scale_emb_or_prj: 指定在计算注意力分数时是否要对嵌入向量或投影层进行缩放。可选值为 'emb'（对嵌入向量进行缩放）或 'prj'（对投影层进行缩放）。默认值为 'prj'。
    """

    def __init__(
            self,
            n_src_vocab,  # 这里是源语言的输入大小：也就是全部待翻译的语言
            n_trg_vocab,  # 这里是指目标语言大小，也就是说对此词之前已经预测出来的词作为输入
            src_pad_idx,
            trg_pad_idx,
            d_word_vec=512,  # 每个词经过位置嵌入后变成512的维度
            d_model=512,  # 模型的维度
            d_inner=2048,
            n_layers=6,
            n_head=8,
            d_k=64,
            d_v=64,
            dropout=0.1,
            n_position=200,
            trg_emb_prj_weight_sharing=True,
            emb_src_trg_weight_sharing=True,
            scale_emb_or_prj='prj'
    ):

        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        # In section 3.4 of paper "Attention Is All You Need", there is such detail:
        # "In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation...
        # In the embedding layers, we multiply those weights by \sqrt{d_model}".
        #
        # Options here: 缩放的两种方式
        #   'emb': multiply \sqrt{d_model} to embedding output
        #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        #   'none': no multiplication

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        # 这里首先根据目标语言的嵌入层和投影层是否进行权重共享进行判断
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False  # 这是一个布尔值，表示是否对嵌入层的输出进行缩放
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False  # 布尔值，表示是否对投影层进行缩放
        self.d_model = d_model

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)  # 初始化，线性层，将模型的维度通过线形层输出转换成词汇表的大小

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)  # 这是使用Xavier初始化方法确保每一层的输出方差应该等于其输入的方差

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'  # 这是为了保证输入和输出进行残差连接

        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight  # 这里是将目标语言的嵌入向量的权重将被共享给投影层 self.trg_word_prj

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight  # 将目标语言的嵌入权重向量和源语言的嵌入层权重向量进行共享


    def forward(self, src_seq, trg_seq):

        # 使用函数 get_pad_mask 获取源语言序列的填充掩码。
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        # get_subsequent_mask 获取的目标语言序列的自回归掩码相结合。自回归掩码的作用是确保在生成目标序列时，模型只能依赖于已经生成的部分内容，而不能依赖于未来的信息
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        seq_logit = self.trg_word_prj(dec_output)  # 共享目标语言的嵌入向量给投影层
        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5  # 这里就是缩放因子 根号（d_model）

        return seq_logit.view(-1, seq_logit.size(2))  # 这里是将投影输出seq_logit进行视图重塑，将其变形成为一个二维张量。 将其中的第第三维变成第二维，然后自动计算原有的维度作为第一维
        # 第一个维度的大小会被计算为：总元素数量 / 第二个维度的大小
