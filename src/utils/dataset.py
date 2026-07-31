import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import utils.tools as tools

class UCFDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict, normal: bool = False, return_path: bool = False):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.normal = normal
        self.return_path = return_path
        normalized_labels = self.df['label'].astype(str).str.strip().str.lower()
        valid_labels = {str(label).strip().lower() for label in label_map}
        invalid_labels = sorted(set(normalized_labels) - valid_labels)
        if invalid_labels:
            raise ValueError(f"unsupported labels in {file_path}: {invalid_labels}")
        if normal == True and test_mode == False:
            self.df = self.df.loc[normalized_labels == 'normal']
            self.df = self.df.reset_index()
        elif test_mode == False:
            self.df = self.df.loc[normalized_labels != 'normal']
            self.df = self.df.reset_index()
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        clip_path = self.df.loc[index]['path']
        clip_feature = np.load(clip_path)
        if self.test_mode == False:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']
        if self.return_path:
            return clip_feature, clip_label, clip_length, clip_path
        return clip_feature, clip_label, clip_length
