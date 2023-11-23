from collections import OrderedDict
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class Config(object):
    device = torch.device('cuda')
    num_class = [15751, 3550, 3515]
    task_type = ['age', 'gender']
    warmup_epoch = 5
    model_name = "testmodel"
    epoch = 500
    

    # Basic setting
    Batch_Size = 128
    length_his = 128
    emb_dim = 128
    user_dim = 128

    # op
    lr = 1e-4
    wd = 1e-4

    # Transformer
    num_translayer = 1
    num_encoder = 2
    num_head = 8
    ffn_dim = emb_dim * 2
    atten_dropout_rate = 0.01
    ffn_dropout_rate = 0.01

    #has_adapter
    has_adapter = False

    #CNN
    window_size = 8
    pool_size = 64 #采样后维度

    # finetune
    dropout_rate = 0.01
    savePath = 'savepath/'

    # checkpoint
    checkpoint = False
    checkpointpath = ''
