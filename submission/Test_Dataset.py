import torch
import numpy as np

from torch.utils import data

class Test_Dataset(data.Dataset):
    def __init__(self, feature_data_path):
        self.feature_data = np.load(feature_data_path, fix_imports=True, encoding='bytes')

    def __len__(self):
        return len(self.feature_data)

    def __getitem__(self, item):
        return torch.Tensor(self.feature_data[item])
