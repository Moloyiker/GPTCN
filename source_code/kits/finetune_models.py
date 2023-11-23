import torch
from torch import nn
from models import DDNN_lstm

class DDNN_lstm_finetune(nn.Module):
    def __init__(self, config, device):
        super(DDNN_lstm_finetune, self).__init__()
        self.config = config
        self.device = device
        self.task_embedding = nn.Embedding(1, config.emb_dim)

        self.DDHCN = DDNN_lstm(config, device).to(device)

        self.fintune = nn.Sequential(
            nn.Linear(config.user_dim, config.user_dim // 2),
            nn.ReLU(),
            #nn.Dropout(config.dropout_rate),
            nn.Linear(config.user_dim // 2, config.user_dim // 4),
            nn.ReLU(),
            #nn.Dropout(config.dropout_rate),
            nn.Linear(config.user_dim // 4, 5),
            nn.Softmax()
        )

    def forward(self, input_data):
        task_emb = self.task_embedding(torch.tensor([0]).to(self.device))
        model_outputs = self.DDHCN(input_data, task_emb)
        user_embeddings = model_outputs[0]
        #print(user_embeddings.shape)
        logits_classifier = self.fintune(user_embeddings)
        model_outputs.append(logits_classifier)
        return model_outputs