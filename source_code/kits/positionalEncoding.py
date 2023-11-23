import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
import copy


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        # 位置编码器类的初始化函数,共有三个参数,分别是d_model:词嵌入维度,  dropout:置0比率,max_len:每个句子的最大长度

        super(PositionalEncoding, self).__init__()
        # 实例化nn中预定义的Dropout层,并将dropout传入其中,获得对象
        self.dropout = nn.Dropout(p=dropout)
        # 初始化一个位置编码矩阵,它是一个e阵,矩阵的大小是max_len x dmodel.
        pe = torch.zeros(max_len, d_model)
        '''
        初始化一个绝对位置矩阵,在我们这里,词汇的绝对位置就是用它的索引去表示。
        所以我们首先使用arange方法获得一个连续自然数向量,然后再使用unsqueeze方法拓展
        又因为参数传的是1,代表矩阵拓展的位置,会使向量变成一个max_len x 1的矩阵,
        '''
        position = torch.arange(0, max_len).unsqueeze(1)
        '''
        绝对位置矩阵初始化之后,接下来就是考虑如何将这些位置信息加入到位置编码矩阵中,
        最简单思路就是先将max_len * 1的绝对位置矩阵(position),变换成nax_len x d_model形状,
        要做这种矩阵变换,就需要一个1xd_model形状的变换矩阵div_tern,
        这个变换矩还希望它能够将自然数的绝对位置编码缩放成足够小的数字,有助于在之后的梯度下降过程的收敛
        首先使用arange获得一个自然数矩阵,但是细心的同学们会发现,我们这里并没有按照预计一样初始化一个1*d_model
        而是有了一个跳跃,???????????????????????
        只初始化了一半即1xd_model/2的矩阵。为什么是一半呢,其实这我们可以把它看作是初始化了两次,而每次初始化的变换矩阵会做不同的处理,
        第一次初始化再正弦波上,第二次在余弦波上
        并把这两个矩阵分别填充在位置编码矩阵的偶数和奇数位置上,组成最终的位置编码矩阵
        '''
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * -(math. log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 二维阔三维
        pe = pe.unsqueeze(0)

        # 将位置编码注册成模型的buffer,不是模型的超参数,不会被优化器更新
        # 注册成buffer后我们就可以在模型保存后重新加载的时候,将这个位置编码器和模型参数
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x:代表文本序社的词嵌入表示
        # 首先明确pe的编码太长了,将第二个维度,也就是max_len对应的那个维度缩小
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


def create_1d_absolute_sin_cos_embedding(pos_len, dim):
    assert dim % 2 == 0, "wrong dimension!"
    position_emb = torch.zeros(pos_len, dim, dtype=torch.float)
    # i矩阵
    i_matrix = torch.arange(dim//2, dtype=torch.float)
    i_matrix /= dim / 2
    i_matrix = torch.pow(10000, i_matrix)
    i_matrix = 1 / i_matrix
    i_matrix = i_matrix.to(torch.long)
    # pos矩阵
    pos_vec = torch.arange(pos_len).to(torch.long)
    # 矩阵相乘，pos变成列向量，i_matrix变成行向量
    out = pos_vec[:, None] @ i_matrix[None, :]
    # 奇/偶数列
    emb_cos = torch.cos(out)
    emb_sin = torch.sin(out)
    # 赋值
    position_emb[:, 0::2] = emb_sin
    position_emb[:, 1::2] = emb_cos
    return position_emb
