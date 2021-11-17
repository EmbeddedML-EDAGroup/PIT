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

class GoogleSpeechCommandsDataLoader(BaseDataLoader):
    """
    GoogleSpeechCommands dataset loading and pre-processing
    """
    def __init__(self, data_dir, batch_size, shuffle=True, set_='train', validation_split=0.0, num_workers=0):
        
        self.data_dir = data_dir
        self.dataset = GoogleSpeechCommandsDataset(data_dir, set_)
        if set_ == 'test':
            super(GoogleSpeechCommandsDataLoader, self).__init__(self.dataset, self.dataset.__len__(), shuffle, validation_split, num_workers)
        else:
            super(GoogleSpeechCommandsDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
    
class GoogleSpeechCommandsDataset(Dataset):
    """
    GoogleSpeechCommands dataset. The data are extracted from the 'train.pkl' 
    or 'valid.pkl' or 'test.pkl' file present in the path passed as data_dir argument.

    :param data_dir: absolute path of the directory containing the dataset
    :param set_: specific dataset to be loaded. Could be 'train', 'eval' or 'test'
    """
    def __init__(self, data_dir, set_='train'):
        super(GoogleSpeechCommandsDataset, self).__init__()

        self.data_dir = Path(data_dir)

        if set_ == 'train':
            if (self.data_dir / 'train.pkl').exists():
                with open(self.data_dir / 'train.pkl', 'rb') as f:
                    self.data = pickle.load(f, encoding='latin1')
                self.X = self.data['X']
                self.y = self.data['y']
            else:
                raise ValueError('train.pkl does not exist')
        elif set_ == 'valid':
            if (self.data_dir / 'valid.pkl').exists():
                with open(self.data_dir / 'valid.pkl', 'rb') as f:
                    self.data = pickle.load(f, encoding='latin1')
                self.X = self.data['X']
                self.y = self.data['y']
            else:
                raise ValueError('valid.pkl does not exist')
        elif set_ == 'test':
            if (self.data_dir / 'test.pkl').exists():
                with open(self.data_dir / 'test.pkl', 'rb') as f:
                    self.data = pickle.load(f, encoding='latin1')
                self.X = self.data['X']
                self.y = self.data['y']
            else:
                raise ValueError('test.pkl does not exist')
        else:
            raise ValueError("Possible 'set_' values are 'train' or 'valid' or 'test'")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
                'data': self.X[idx],
                'target': self.y[idx]
                }

        return sample



