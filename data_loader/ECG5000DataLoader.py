#*----------------------------------------------------------------------------*
#* Copyright (C) 2021 Politecnico di Torino, Italy                            *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Author:  Matteo Risso <matteo.risso@polito.it>                             *
#*----------------------------------------------------------------------------*

import torch
from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
from pathlib import Path
import pickle
import numpy as np
from scipy.io import loadmat
import random
import pandas as pd
import os
import pdb

class ECG5000DataLoader(BaseDataLoader):
    """
    ECG5000 dataset loading and pre-processing
    """
    def __init__(self, data_dir, batch_size, shuffle=True, set_='train', validation_split=0.0, num_workers=0, sampler=None):
        
        self.data_dir = data_dir
        self.dataset = ECG5000Dataset(data_dir, set_)
        self.sampler = sampler
        
        if set_ == 'test':
            super(ECG5000DataLoader, self).__init__(self.dataset, self.dataset.__len__(), shuffle, validation_split, num_workers)
        else:
            super(ECG5000DataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, sampler=self.sampler)

class ECG5000Dataset(Dataset):
    """
    ECG5000 dataset. The dataset is extracted from the 'ECG5000_TEST.txt' or 'ECG5000_TRAIN.txt' file present in the path passed as data_dir argumenti.
    The data are organized as a dictionary and saved in a '.pkl' file.

    :param data_dir: absolute path of the directory containing the dataset
    :param set_: specific dataset to be loaded. Could be 'train', 'eval' or 'test'
    """
    def __init__(self, data_dir, set_='train'):
        super(ECG5000Dataset, self).__init__()

        self.data_dir = Path(data_dir)

        # Always load train data in order to obtain normalization info 
        if os.path.exists(self.data_dir / 'ecg5000_train.pkl'):
            with open(self.data_dir / 'ecg5000_train.pkl', 'rb') as f:
                self.data = pickle.load(f, encoding='latin1')
            self.X = self.data['X_train']
            self.y = self.data['y_train']
        else:
            self.X, self.y = self._prepare_data(set_)
        
        self.min = self.X.min()
        self.max = self.X.max()
        
        if set_ == 'test' or set_ == 'valid':
            if os.path.exists(self.data_dir / 'ecg5000_test.pkl'):
                with open(self.data_dir / 'ecg5000_test.pkl', 'rb') as f:
                    self.data = pickle.load(f, encoding='latin1')
                self.X = self.data['X_test']
                self.y = self.data['y_test']
            else:
                self.X, self.y = self._prepare_data(set_)
        elif set_ == 'train':
            pass
        else:
            raise ValueError("Possible 'set_' values are 'train' or 'test'")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
                #'data': self.X[idx],
                'data': (self.X[idx] - self.min) / (self.max - self.min),
                'target': self.y[idx]
                }

        return sample

    def _prepare_data(self, set_):
        dataset = np.loadtxt(self.data_dir / ('ECG5000_' + set_ + '.txt'))
        y = dataset[:, 0].astype(np.int) - 1
        X = dataset[:, 1:].astype(np.float32)
       
        data = {
            'X_'+set_: X,
            'y_'+set_: y
            }
        
        with open(self.data_dir / ('ecg5000_' + set_ + '.pkl'), 'wb') as f:
            pickle.dump(data, f)

        return X, y

    

