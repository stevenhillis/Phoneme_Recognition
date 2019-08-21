import torch
import numpy as np

from torch.utils import data

class Train_Dataset(data.Dataset):
    # feature_data: (frames, time_step, 40) with time_step variable
    # label_data: (frames, frequencies) with frequencies variable
    def __init__(self, feature_data_path, label_data):
        self.feature_data = np.load(feature_data_path, fix_imports=True, encoding='bytes')
        self.label_data = np.load(label_data, fix_imports=True, encoding='bytes')

    def __len__(self):
        return len(self.feature_data)

    def __getitem__(self, item):
        return torch.Tensor(self.feature_data[item]), torch.Tensor(self.label_data[item])
