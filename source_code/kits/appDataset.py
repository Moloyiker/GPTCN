import pandas as pd
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Lambda, Compose

def _string_to_list(length_his, string, padding):
    list = [int(x) for x in string.split(',')]
    list = np.asarray(list)
    tensor = torch.from_numpy(list)
    tensor = torch.concat([torch.tile(torch.Tensor([padding]), (max(length_his-tensor.shape[0], 0), )),
                           tensor[-length_his:]], axis=0)
    return tensor


def _string_to_onehot(length_his, string, num_class):
    list = [int(x) for x in string.split(',')]
    list = np.asarray(list)
    tensor = torch.from_numpy(list) + 1
    tensor = torch.concat([torch.tile(torch.Tensor([0]), (max(length_his-tensor.shape[0], 0), )),
                           tensor[-length_his:]], axis=0).to(torch.int64)
    one_hot = F.one_hot(tensor, num_class + 1)[:, 1:]
    return one_hot


class AppInstallDataset(Dataset):
    def __init__(self, file, num_class, length_his=64):
        super().__init__()
        self.install_data = pd.read_csv(file, encoding='utf-8-sig', sep='|')
        self.input_keys = ['uid', 'new_soft_list','new_time_list', 'new_sep_list', 'new_mask_list']
        self.label_keys = ['age', 'gender']

        self.stringList_transform = Lambda(
            lambda x: _string_to_list(length_his, x, -1))
        self.stringList_transform_forsep = Lambda(
            lambda x: _string_to_list(length_his, x, 0))
        self.stringList_apponeHot_transform = Lambda(
            lambda x: _string_to_onehot(length_his, x, num_class[0]))
        self.stringList_timeoneHot_transform = Lambda(
            lambda x: _string_to_onehot(length_his, x, num_class[1]))
        self.stringList_seponeHot_transform = Lambda(
            lambda x: _string_to_onehot(length_his, x, num_class[2]))
        
        self.mask_transform = Lambda(lambda x: torch.greater_equal(
            x, torch.tile(torch.Tensor([0]), (length_his,))).int())
        self.int_transform = Lambda(lambda x: torch.Tensor([x]).int())

    def __len__(self):
        return len(self.install_data)

    def __getitem__(self, idx):
        device = torch.device('cuda')
        input_data = dict.fromkeys(self.input_keys)
        label = dict.fromkeys(self.label_keys)

        input_data['uid'] = torch.tensor(self.install_data.iloc[idx, 0])
        input_data['new_soft_list'] = self.stringList_transform(self.install_data.iloc[idx, 1]).int()
        input_data['new_time_list'] = self.stringList_transform(self.install_data.iloc[idx, 2]).int()
        input_data['new_sep_list'] = self.stringList_transform_forsep(self.install_data.iloc[idx, 3]).int()

        input_data['new_mask_list'] = (~(self.mask_transform(input_data['new_time_list']).bool()))

        input_data['new_soft_list_onehot'] = self.stringList_apponeHot_transform(self.install_data.iloc[idx, 1]).int()
        input_data['new_time_list_onehot'] = self.stringList_timeoneHot_transform(self.install_data.iloc[idx, 2]).int()
        input_data['new_sep_list_onehot'] = self.stringList_seponeHot_transform(self.install_data.iloc[idx, 3]).int()

        label['age'] = torch.tensor(self.install_data.iloc[idx, 4]).to(device)
        label['gender'] = torch.tensor(self.install_data.iloc[idx, 5]).to(device)

        input_data['uid'] = input_data['uid'].to(device)
        input_data['new_soft_list'] = input_data['new_soft_list'].to(device)
        input_data['new_time_list'] = input_data['new_time_list'].to(device)
        input_data['new_sep_list'] = input_data['new_sep_list'].to(device)
        input_data['new_mask_list'] = input_data['new_mask_list'].to(device)
        input_data['new_soft_list_onehot'] = input_data['new_soft_list_onehot'].to(device)
        input_data['new_time_list_onehot'] = input_data['new_time_list_onehot'].to(device)
        input_data['new_sep_list_onehot'] = input_data['new_sep_list_onehot'].to(device)
        return input_data, label


