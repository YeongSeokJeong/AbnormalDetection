from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Dict
import os.path
from PIL import Image
from hydra.utils import instantiate
import torchvision
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch


class BaseDataset(Dataset):
    def __init__(self, X, Y, aug_functions: Dict, data_root, is_test=False):
        super(BaseDataset, self).__init__()
        self.is_test = is_test
        self.data_root = data_root
        self.to_tf = torchvision.transforms.ToTensor()
        aug_functions = [augmentation for key, augmentation in aug_functions.items()]
        aug_functions.append(
            torchvision.transforms.Resize([224,224])
        )
        self.transforms = torchvision.transforms.Compose(aug_functions)

        self.img_paths = self.get_img_paths(X)

        self.label = Y

        self.num_img = len(self.img_paths)

    def get_img_paths(self, file_names):
        paths = [os.path.join(self.data_root, file_name) for file_name in file_names]
        return paths

    def load_img(self, path):
        img = Image.open(path).convert('RGB')
        img = self.to_tf(img)/255
        return img

    def __len__(self):
        return self.num_img

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = self.load_img(path)
        img = self.transforms(img)
        if not self.is_test:
            label = self.label[idx]
            return img, torch.tensor(label, dtype=torch.int64)
        else:
            return img


class BaseDataLoader(object):
    def __init__(self, data_root, train_augs, valid_augs, batch_size, num_workers=0, fold_num: int = 0, seed=255, is_test=False):
        super(BaseDataLoader, self).__init__()
        assert fold_num < 5, ValueError
        self.data_root = data_root
        self.label_idx = self.get_label_idx()
        self.is_test = is_test

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_augs = train_augs
        self.valid_augs = valid_augs

        if not is_test:
            dataframe = pd.read_csv(f'{data_root}/train_df.csv')
            X = dataframe['file_name'].to_numpy()
            Y = np.array([self.label_idx[label] for label in dataframe['label'].to_list()])
            stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for i, (train_index, valid_index) in enumerate(stratified_kfold.split(X, Y)):
                if i == fold_num:
                    self.train_x, self.valid_x = X[train_index], X[valid_index]
                    self.train_y, self.valid_y = Y[train_index], Y[valid_index]
                    self.train_x = [os.path.join('train', file_name) for file_name in self.train_x]
                    self.valid_x = [os.path.join('train', file_name) for file_name in self.valid_x]

                    break
        else:
            dataframe = pd.read_csv(f'{data_root}/test_df.csv')
            self.X = dataframe['file_name'].to_numpy()

    def get_label_idx(self):
        label_df = pd.read_csv(f'{self.data_root}/train_df.csv')
        label_idx = {label: i for i, label in enumerate(sorted(list(set(label_df['label']))))}
        return label_idx

    def get_train_valid_dataloaders(self):
        train_dataset = BaseDataset(self.train_x, self.train_y, self.train_augs, self.data_root)
        valid_dataset = BaseDataset(self.valid_x, self.valid_y, self.valid_augs, self.data_root)

        return DataLoader(train_dataset, self.batch_size, True, num_workers=self.num_workers),\
               DataLoader(valid_dataset, self.batch_size, False, num_workers=self.num_workers)

    def get_test_dataloaders(self):
        test_dataset = BaseDataset(self.X, None, self.train_augs, self.data_root, self.is_test)
        return DataLoader(test_dataset, self.batch_size, False, num_workers=self.num_workers)
