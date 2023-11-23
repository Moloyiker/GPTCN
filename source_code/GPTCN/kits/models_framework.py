from distutils.log import debug
from hashlib import new
from this import d
from matplotlib.pyplot import cla
import torch
from torch import nn
import torch.nn.functional as F
from kits.attention import Gated_MultiHeaded_self_Attention, GMSA

class CNN(nn.Module):
    def __init__(self, config, out_channels):  # output_dim表示CNN输出通道数
        super(CNN, self).__init__()
        # 1个输入通道，out_channels个输出通道，过滤器大小为window_size*word_dim
        self.config = config
        self.new_seq_len = config.length_his - config.window_size + 1
        self.window_size = config.window_size
        self.out_channels = out_channels
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=(config.window_size, config.emb_dim)),
            nn.LeakyReLU()
        )
        #self.maxpool = nn.AdaptiveMaxPool1d(config.pool_size) #采样 输出64
        self.pool = nn.AdaptiveAvgPool1d(config.pool_size)
        self.dnn = nn.Sequential(
            nn.Linear(self.out_channels * config.pool_size, self.out_channels),
            nn.LeakyReLU()
        )

    # 输入x为batch组文本，长度为seq_len，词向量长度为word_dim，维度为batch*seq_len*word_dim
    # 输出res为所有文本向量，每个向量的维度为out_channels
    def forward(self, x):
        in_size = x.size(0)
        x_unsqueeze = x.unsqueeze(1)  # 变成单通道，结果维度为batch*1*seq_len*word_dim 128 1 128 128
        x_cnn = self.cnn(x_unsqueeze)  # CNN，结果维度为batch*out_channels*new_seq_len*1
        x_cnn_result = x_cnn.squeeze(3)  # 删除最后一维，结果维度为batch*out_channels*new_seq_len
        #print(x_cnn_result.shape)
        res = self.pool(x_cnn_result) #(batch, out_channels, new_seq_len / 2)
        #print(res.shape)
        res = res.view(in_size, -1)
        res = self.dnn(res)
        return res

class CNN_Decoder(nn.Module):
    def __init__(self, config, out_channels):  # output_dim表示CNN输出通道数
        super(CNN_Decoder, self).__init__()
        self.config = config
        self.out_channels = out_channels
        # 1个输入通道，out_channels个输出通道，过滤器大小为window_size*word_dim
        self.new_seq_len = config.length_his - config.window_size + 1

        self.dnn = nn.Sequential(
            nn.Linear(self.out_channels, self.out_channels * config.pool_size),
            nn.LeakyReLU()
        )
        self.upsampling = nn.Upsample(scale_factor=self.new_seq_len / config.pool_size, mode='linear')
        self.cnn_transpose = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.out_channels, out_channels=1, kernel_size=(config.window_size, config.emb_dim)),
            nn.LeakyReLU()
        )

    def forward(self, x):
        #in_size = x.size(0)
        x = self.dnn(x) #(out_channels * 64)
        x = x.view(-1, self.out_channels, self.config.pool_size)
        x = self.upsampling(x) #(batch, out_channels, new_seq_len)
        x = x.unsqueeze(3) #(batch, out_channels, new_seq_len, 1)
        x = self.cnn_transpose(x)
        x = x.squeeze(1)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.config = config
        self.layer_norm_epsilon = 1e-6
        self.mutihead_attn = nn.MultiheadAttention(
            config.emb_dim, config.num_head, batch_first=True, dropout=config.atten_dropout_rate)
        self.ffn = nn.Sequential(
            nn.Linear(config.emb_dim, config.ffn_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.ffn_dropout_rate),
            nn.Linear(config.ffn_dim, config.emb_dim)
        )
        self.layernorm1 = nn.LayerNorm(
            config.emb_dim, eps=self.layer_norm_epsilon)
        self.layernorm2 = nn. LayerNorm(
            config.emb_dim, eps=self.layer_norm_epsilon)

    def forward(self, input, newmask, taskemb):
        # print(newmask)
        # print(type(newmask))
        y = self.mutihead_attn(input, input, input, key_padding_mask=newmask, need_weights=False)[0] + input
        y = self.layernorm1(y)
        # attn_output = y
        y = self.ffn(y) + y
        y = self.layernorm2(y)
        return y

class GateTransformerEncoder(nn.Module):
    def __init__(self, config):
        super(GateTransformerEncoder, self).__init__()
        self.config = config
        self.layer_norm_epsilon = 1e-6
        self.mutihead_attn = GMSA(
            config.emb_dim, config.num_head, batch_first=True, dropout=config.atten_dropout_rate)
        self.ffn = nn.Sequential(
            nn.Linear(config.emb_dim, config.ffn_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.ffn_dropout_rate),
            nn.Linear(config.ffn_dim, config.emb_dim)
        )
        self.layernorm1 = nn.LayerNorm(
            config.emb_dim, eps=self.layer_norm_epsilon)
        self.layernorm2 = nn.LayerNorm(
            config.emb_dim, eps=self.layer_norm_epsilon)

    def forward(self, input, newmask):
        # print('beforeMSA_input',input.size())
        # print('beforeMSA_newmask',newmask.size())
        y = self.mutihead_attn(input, input, input, newmask)[0] + input
        y = self.layernorm1(y)
        # attn_output = y
        y = self.ffn(y) + y
        y = self.layernorm2(y)
        return y


class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super(TransformerDecoder, self).__init__()
        self.config = config
        self.layer_norm_epsilon = 1e-6
        self.mutihead_attn = nn.MultiheadAttention(config.emb_dim, config.num_head, batch_first=True, dropout=config.atten_dropout_rate)
        #self.mutihead_selfattn = nn.MultiheadAttention(config.emb_dim, config.num_head, batch_first=True, dropout=config.atten_dropout_rate)
        
        self.ffn = nn.Sequential(
            nn.Linear(config.emb_dim, config.ffn_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.ffn_dropout_rate),
            nn.Linear(config.ffn_dim, config.emb_dim),
        )
        self.layernorm1 = nn.LayerNorm(
            config.emb_dim, eps=self.layer_norm_epsilon)
        self.layernorm2 = nn.LayerNorm(
            config.emb_dim, eps=self.layer_norm_epsilon)

    def forward(self, input, input_encoder, newmask):
        y = self.mutihead_attn(input, input_encoder, input_encoder,
                               key_padding_mask=newmask, need_weights=False)[0]
        y = self.layernorm1(y)

        y = self.ffn(y) + y
        y = self.layernorm2(y)

        return y

class DDHCN(nn.Module):
    def __init__(self, config, device):
        super(DDHCN, self).__init__()
        self.config = config
        self.batch_size = config.Batch_Size
        self.num_translayer = config.num_translayer
        self.num_encoder = config.num_encoder
        self.emb_dim = config.emb_dim
        self.length_his = config.length_his
        self.user_dim = config.user_dim
        self.device = device
        self.has_adapter = config.has_adapter
        self.app_embeddings = nn.Embedding(config.num_class[0] + 1, self.emb_dim, padding_idx=0)
        self.date_embeddings = nn.Embedding(config.num_class[1] + 1, self.emb_dim, padding_idx=0)
        self.sep_embeddings = nn.Embedding(config.num_class[2] + 1, self.emb_dim, padding_idx=0)
        
        # #self.sep_fc = nn.Linear(self.length_his * self.emb_dim, self.emb_dim)
        # if self.has_adapter:
        #     if self.num_translayer > self.num_encoder:
        #         num_layer = self.num_translayer
        #     else:
        #         num_layer = self.num_encoder
        #     self.hyperNet = HyperNetController(config, self.device, num_layer)

        self.EncoderBlock = nn.ModuleList(
            [TransformerEncoder(config) for i in range(self.num_encoder)]
        )
        self.cnn_encoder = CNN(config, self.user_dim)
        self.cnn_decoder_app = CNN_Decoder(config, self.user_dim)
        self.cnn_decoder_time = CNN_Decoder(config, self.user_dim)
        
        self.DecoderBlock = nn.ModuleList(
            [TransformerDecoder(config) for i in range(self.num_translayer * 2)]
        )

    def forward(self, input_data, task_emb):
        model_outpus = []

        newapp = input_data['new_soft_list'] + 1
        newdate = input_data['new_time_list'] + 1
        newsep = input_data['new_sep_list'] 
        newmask = input_data['new_mask_list']


        new_app_emb = self.app_embeddings(newapp)
        # (batch, length, emb_dim) (T0, T1, T2,..., TN)
        new_time_emb = self.date_embeddings(newdate)
        new_sep_emb = self.sep_embeddings(newsep)
        # t_mat1 = new_time_emb.unsqueeze(1).repeat(1, self.length_his, 1, 1) #(N * T0, T1, T2, ..., TN)
        #t_mat2 = new_time_emb.unsqueeze(2).repeat(1, 1, self.length_his, 1)

        # new_sep_emb = t_mat2 - t_mat1 #(batch, length_his, length_his, emb_size) ->(batch, length_his, length_his*emb_size)
        #new_sep_emb = self.sep_fc(new_sep_emb.flatten(2))
        # print(new_app_emb.size()) torch.Size([128, 128, 128])
        new_emb = new_app_emb + new_time_emb + new_sep_emb
        #new_emb = self.pos_emb(new_emb)
        
        # newmask torch.Size([128, 128])
        # y = x[None,:, :].repeat(128, 1,1)
        # newmask = newmask[None,:, :].repeat(128, 1,1)

        # print(new_emb.size())  #torch.Size([128, 128, 128])
        # print(type(newmask))
        # print(newmask.size())  #torch.Size([128, 128, 128])

        # n = 0
        # assert n != 0, 'n is zero!'


        # Encoder
        for trans_layer in range(self.num_encoder):
            new_emb = self.EncoderBlock[trans_layer](new_emb, newmask, None)

        # Bottlneck layer
        user_embeddings = self.cnn_encoder(new_emb)
        model_outpus.append(user_embeddings)
        use_emb_app = self.cnn_decoder_app(user_embeddings)
        use_emb_time = self.cnn_decoder_time(user_embeddings)

        #pos_emb = self.pos_emb.weight[:self.length_his, :].view(1, self.length_his, self.emb_dim)
        # print(pos_emb.shape)

        # Decoder app
        new_reapp_emb = new_time_emb + new_sep_emb
        for trans_layer in range(self.num_translayer):
            # if self.has_adapter:
            #     new_reapp_emb = self.DecoderBlock[trans_layer](
            #         new_reapp_emb, use_emb_app, newmask, self.hyperNet(task_emb, trans_layer, 1))
            # else:
            new_reapp_emb = self.DecoderBlock[trans_layer](
                new_reapp_emb, use_emb_app, newmask)
        app_weights = torch.transpose(self.app_embeddings.weight[1:], 0, 1)
        logits_new_app = torch.einsum(
            'ijk,kl->ijl', new_reapp_emb, app_weights)
        model_outpus.append(logits_new_app)

        # Decoder time
        new_retime_emb = new_app_emb + new_time_emb
        for trans_layer in range(self.num_translayer):
            # if self.has_adapter:
            #     new_retime_emb = self.DecoderBlock[trans_layer+self.num_translayer](
            #         new_retime_emb, use_emb_time, newmask, self.hyperNet(task_emb, trans_layer, 2))
            # else:
            new_retime_emb = self.DecoderBlock[trans_layer+self.num_translayer](
                    new_retime_emb, use_emb_time, newmask)
        time_weights = torch.transpose(self.sep_embeddings.weight[1:], 0, 1)
        logits_new_time = torch.einsum(
            'ijk,kl->ijl', new_retime_emb, time_weights)
        model_outpus.append(logits_new_time)

        # user_embeddings
        # user_embeddings1 = f1(x)
        # user_embeddings2 = f2(x)

        # pre1  = dnn1(user_embeddings1)
        # pre2  = dnn2(user_embeddings2)

        return model_outpus



# mix-attention
class seqFusionAttentionNoBiasDiff(nn.Module):
    '''
    N_batch x S_seq x D_dim
    '''
    def __init__(self,qDim,kDim=None,vDim=None,outDim=None,cDim=None,headNum=4,
                 dropout_rate=0.15,device=None,dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(seqFusionAttentionNoBiasDiff,self).__init__()
        if kDim is None:
            kDim = qDim
        if vDim is None:
            vDim = kDim
        
        if outDim is None:
            outDim = vDim
        if cDim is None:
            cDim = round(qDim/headNum)
        self.register_buffer('QTrans',None)
        self.QTrans = nn.Parameter(torch.empty((qDim, cDim*headNum), **factory_kwargs), requires_grad = True)
                
        self.register_buffer('KTrans',None)
        self.KTrans = nn.Parameter(torch.empty((kDim, cDim*headNum), **factory_kwargs), requires_grad = True)
        
        self.register_buffer('VTrans',None)
        self.VTrans = nn.Parameter(torch.empty((vDim, cDim*headNum), **factory_kwargs), requires_grad = True)

        # self.register_buffer('GTrans',None)
        # self.GTrans = nn.Parameter(torch.empty((vDim, cDim*headNum), **factory_kwargs), requires_grad = True)
        
        # self.register_buffer('GTransBias',None)
        # self.GTransBias = nn.Parameter(torch.empty((headNum, 1), **factory_kwargs), requires_grad = True)
        
        self.GTrans = nn.Linear(vDim, cDim*headNum)
        
        self.outTrans = nn.Linear(cDim*headNum,outDim,bias=True)
        self.softMaxLayer = nn.Softmax(dim=-1)
        # self.softMaxLayer = nn.LogSoftmax(dim=-1)
        self.cDim = cDim
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p = self.dropout_rate)
        self.headNum = headNum
        self.device = device
        self._reset_parameters()
        self.sigmoid = torch.nn.Sigmoid()
        if device == 'cpu':
            self.type = torch.float
        else:
            self.type = torch.float
        self.mixAtt = []
        
    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.QTrans)
        torch.nn.init.xavier_uniform_(self.KTrans)
        torch.nn.init.xavier_uniform_(self.VTrans)
        # torch.nn.init.xavier_uniform_(self.BiasTrans1)
        # torch.nn.init.xavier_uniform_(self.BiasTrans2)
        torch.nn.init.zeros_(self.GTrans.weight)
        torch.nn.init.ones_(self.GTrans.bias)
        torch.nn.init.zeros_(self.outTrans.weight)
        torch.nn.init.zeros_(self.outTrans.bias)
        
    def forward(self,Qin,Kin,Vin,seqMask=None):
        batchSize = Qin.size(0)
        # seqLen = Qin.size(1)
        Q = torch.einsum('nsi,ij->nsj',Qin,self.QTrans).view([batchSize,Qin.size(1),self.headNum,self.cDim]).permute([2,0,1,3]) #HxNxSxDc
        K = torch.einsum('nsi,ij->nsj',Kin,self.KTrans).view([batchSize,Kin.size(1),self.headNum,self.cDim]).permute([2,0,1,3]) #HxNxSxDc
        V = torch.einsum('nsi,ij->nsj',Vin,self.VTrans).view([batchSize,Vin.size(1),self.headNum,self.cDim]).permute([2,0,1,3]) #HxNxSxDc
        
        
        
        # G = torch.einsum('nsi,ij->nsj',Vin,self.GTrans).permute([2,0,1]) + self.GTransBias.view([Q.size(0),1,1]).expand(-1,Q.size(1),1)#HxNxS
        G = self.GTrans(Vin).view([batchSize,Vin.size(1),self.headNum,self.cDim]).permute([2,0,1,3]) #HxNxSxDc
        G = self.sigmoid(G)
        # print(G.detach().cpu().numpy())
        
       
        A = torch.einsum('hnsi,hnti->hnst',Q,K)/torch.sqrt(torch.tensor(self.cDim,device=self.device,dtype=self.type)) #HxNxSxS
        if not seqMask is None:
            # seqMask = seqMask.view([1,seqMask.size(0),1,seqMask.size(1)]).expand(self.headNum,-1,seqLen,-1) #HxNxSxS
            seqMask = torch.einsum('ns,nt->nst',seqMask.float(),seqMask.float()).bool().view([1,batchSize,Qin.size(1),Kin.size(1)]).expand([self.headNum,-1,-1,-1])
            A.masked_fill_(seqMask,float('-inf'))
        A = self.softMaxLayer(A) #HxNxSxS
        ## checkpoint
        # self.mixAtt.append([item.cpu().detach().numpy() for item in A])
        if self.dropout_rate > 0:
            A = self.dropout(A)
        # A = self.softMaxLayer(torch.einsum('hnsi,hnti->nst',Q,K)/torch.sqrt(torch.tensor(self.cDim,device=self.device,dtype=self.type)) + Bias) #HxNxSxS
        # V = torch.einsum('hnst,hnti->hnsi',A,V).permute([1,2,0,3]) #NxSxHxDc
        # V = torch.einsum('hnst,hnti->nshi',A,V)#NxSxHxDc
        V = torch.einsum('hnst,hnti,hnti->nshi',A,V,G)#NxSxHxDc
        out = self.outTrans(V.reshape([V.size(0),V.size(1),-1]))

        return out
    
class FeedForward(nn.Module):
    def __init__(self, dimIn,dropout_rate,alpha=8 ,zeroLL=True, withFF=True):
        
        super(FeedForward, self).__init__()
        self.layerNorm = nn.LayerNorm(dimIn)
        self.layerNorm1 = nn.LayerNorm(dimIn*alpha)
        self.fc1 = nn.Linear( dimIn,dimIn*alpha)
      
        self.fc2 = nn.Linear( dimIn*alpha,dimIn)
        
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p = self.dropout_rate)
        self.withFF = withFF
        self._reset_parameters()
        if zeroLL:
            self.zeroLastLayer()
    
    def _reset_parameters(self):
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.fc2.weight)  
        

    def zeroLastLayer(self):
        torch.nn.init.zeros_(self.fc2.weight) #last output layer
        torch.nn.init.zeros_(self.fc2.bias) #last output layer
        
    def forward(self, x):
        if self.withFF:
            x = self.layerNorm(x)
        # x = self.elu(x)
        # x = self.dropout(x)
        x = self.fc1(x)
        # x = self.layerNorm1(x)
        # x = self.elu(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.layerNorm(x)        
        # x = self.elu(x)
        # x = self.dropout(x)
        
        return x 


class MFR(nn.Module):
    def __init__(self, config, device, drop_p:float = 0.15,):
        super(MFR, self).__init__()
        self.config = config
        self.batch_size = config.Batch_Size
        self.num_translayer = config.num_translayer
        self.num_encoder = config.num_encoder
        self.emb_dim = config.emb_dim
        self.length_his = config.length_his
        self.user_dim = config.user_dim
        self.device = device
        self.has_adapter = config.has_adapter
        self.app_embeddings = nn.Embedding(config.num_class[0] + 1, self.emb_dim, padding_idx=0)
        self.date_embeddings = nn.Embedding(config.num_class[1] + 1, self.emb_dim, padding_idx=0)
        self.sep_embeddings = nn.Embedding(config.num_class[2] + 1, self.emb_dim, padding_idx=0)
        
        # #self.sep_fc = nn.Linear(self.length_his * self.emb_dim, self.emb_dim)
        # if self.has_adapter:
        #     if self.num_translayer > self.num_encoder:
        #         num_layer = self.num_translayer
        #     else:
        #         num_layer = self.num_encoder
        #     self.hyperNet = HyperNetController(config, self.device, num_layer)

        self.EncoderBlock = nn.ModuleList(
            [TransformerEncoder(config) for i in range(self.num_encoder * 2)]
        )
        self.corssAtt = seqFusionAttentionNoBiasDiff(self.emb_dim,dropout_rate=drop_p)

        self.cnn_encoder = CNN(config, self.user_dim)
        self.cnn_decoder_app = CNN_Decoder(config, self.user_dim)
        self.cnn_decoder_time = CNN_Decoder(config, self.user_dim)

        self.DecoderBlock = nn.ModuleList(
            [TransformerDecoder(config) for i in range(self.num_translayer * 2)]
        )
        self.FF = FeedForward(self.emb_dim, drop_p)
    def forward(self, input_data, task_emb):
        model_outpus = []

        newapp = input_data['new_soft_list'] + 1
        newdate = input_data['new_time_list'] + 1
        newsep = input_data['new_sep_list'] 
        newmask = input_data['new_mask_list']


        new_app_emb = self.app_embeddings(newapp)
        # (batch, length, emb_dim) (T0, T1, T2,..., TN)  torch.Size([128, 128, 128])
        new_time_emb = self.date_embeddings(newdate)
        new_sep_emb = self.sep_embeddings(newsep)
        # t_mat1 = new_time_emb.unsqueeze(1).repeat(1, self.length_his, 1, 1) #(N * T0, T1, T2, ..., TN)
        #t_mat2 = new_time_emb.unsqueeze(2).repeat(1, 1, self.length_his, 1)
        # new_emb = self.pos_emb(new_emb)


        item_emb = new_app_emb 
        behavior_emb = new_time_emb + new_sep_emb
        
        # print(new_app_emb.size()) 
        # newmask torch.Size([128, 128])
        # y = x[None,:, :].repeat(128, 1,1)
        # newmask = newmask[None,:, :].repeat(128, 1,1)

        '''
        其中item_emb和behavior_emb被区分对待。
        behavior_emb可以表征用户在item的行为,哪些需要被记忆,哪些需要被抽取;
        item_emb表示Item的特征,反应了用户的兴趣,作为唯一输入。
        '''
        # Encoder
        for trans_layer in range(self.num_encoder):
            new_item_emb = self.EncoderBlock[trans_layer](item_emb, newmask, None)
            new_behavior_emb = self.EncoderBlock[trans_layer+self.num_translayer](behavior_emb, newmask, None)

        # asm_img_corssAtt
        new_emb = self.corssAtt(new_behavior_emb,new_item_emb,new_item_emb)
        new_emb = new_emb + new_item_emb
        new_emb = new_emb + self.FF(new_emb)

        # print('new_emb',new_emb.size())  #torch.Size([128, 128, 128])

        # print(type(newmask))
        # print('newmask',newmask.size())  #torch.Size([128, 128, 128])

        # n = 0
        # assert n != 0, 'n is zero!'


        # Bottlneck layer
        user_embeddings = self.cnn_encoder(new_emb)
        model_outpus.append(user_embeddings)
        use_emb_app = self.cnn_decoder_app(user_embeddings)
        use_emb_time = self.cnn_decoder_time(user_embeddings)

        #pos_emb = self.pos_emb.weight[:self.length_his, :].view(1, self.length_his, self.emb_dim)
        # print(pos_emb.shape)

        # Decoder app
        new_reapp_emb = new_time_emb + new_sep_emb
        for trans_layer in range(self.num_translayer):
            if self.has_adapter:
                new_reapp_emb = self.DecoderBlock[trans_layer](
                    new_reapp_emb, use_emb_app, newmask, self.hyperNet(task_emb, trans_layer, 1))
            else:
                new_reapp_emb = self.DecoderBlock[trans_layer](
                    new_reapp_emb, use_emb_app, newmask)
        
        app_weights = torch.transpose(self.app_embeddings.weight[1:], 0, 1)
        logits_new_app = torch.einsum(
            'ijk,kl->ijl', new_reapp_emb, app_weights)
        model_outpus.append(logits_new_app)

        # Decoder time
        new_retime_emb = new_app_emb + new_time_emb
        for trans_layer in range(self.num_translayer):
            if self.has_adapter:
                new_retime_emb = self.DecoderBlock[trans_layer+self.num_translayer](
                    new_retime_emb, use_emb_time, newmask, self.hyperNet(task_emb, trans_layer, 2))
            else:
                new_retime_emb = self.DecoderBlock[trans_layer+self.num_translayer](
                    new_retime_emb, use_emb_time, newmask)
            
        time_weights = torch.transpose(self.sep_embeddings.weight[1:], 0, 1)
        logits_new_time = torch.einsum(
            'ijk,kl->ijl', new_retime_emb, time_weights)
        model_outpus.append(logits_new_time)

        return model_outpus
    


