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
import scipy.io
import scipy.signal
import random
import pandas as pd
import data_loader.data_augmentation as da
import os
import pdb

np.random.seed(1234)
random.seed(12345)

class NinaProDB1DataLoader(BaseDataLoader):
    """
    NinaProDB1 dataset loading and pre-processing
    """
    def __init__(self, data_dir, batch_size, shuffle=True, subject=None, set_='train', validation_split=0.0, num_workers=0):
        
        self.data_dir = data_dir
        self.dataset = NinaProDB1Dataset(data_dir, batch_size, subject, set_)
        self.batch_size = batch_size
        
        if set_ == 'test':
            super(NinaProDB1DataLoader, self).__init__(self.dataset, None, False, validation_split, num_workers, collate_fn=None)
        else:
            # with batch_size equal to None automatic batching is disabled
            super(NinaProDB1DataLoader, self).__init__(self.dataset, None, False, validation_split, num_workers, collate_fn=None)
    
class NinaProDB1Dataset(Dataset):
    """
    ECG5000 dataset. The dataset is extracted from the 'ECG5000_TEST.txt' or 'ECG5000_TRAIN.txt' file present in the path passed as data_dir argumenti.
    The data are organized as a dictionary and saved in a '.pkl' file.

    :param data_dir: absolute path of the directory containing the dataset
    :param set_: specific dataset to be loaded. Could be 'train', 'eval' or 'test'
    """
    def __init__(self, data_dir, batch_size, subject, set_='train'):
        super(NinaProDB1Dataset, self).__init__()

        self.subject = subject
        self.data_dir = Path(data_dir)
        if subject is not None:
            self.input_directory = ['{}/subject-{:02d}'.format(data_dir, subject)]
        else:
            self.input_directory = ['{}/subject-{:02d}'.format(data_dir, i) for i in range(1, 28)]
        
        self.batch_size = batch_size
        self.dim = (1, 10)
        self.classes = [i for i in range(53)]
        self.n_classes = len(self.classes)
        self._make_class_index()
        
        self.scale_sigma = 0.
        self.window_size = 0
        self.window_step = 0
        self.rotation = 0
        self.rotation_mask = None
        self.permutation = 0
        
        self.data_type = 'rms'
        self.preprocess_function = lpf
        self.preprocess_function_extra = {'fs': 100}
        self.pad_len = None
        self.pad_value = 0.
        self.min_max_norm = False
        self.update_after_epoch = False
        self.label_proc = None
        self.label_proc_extra = None

        if set_ == 'train':
            self.repetitions = [1, 3, 4, 6, 8, 9, 10]
            self.sample_weight = True
            self.n_reps = len(self.repetitions) 
            self.shuffle = True
            self.noise_snr_db = 25
            self.time_warping = 0.2
            self.mag_warping = 0.2
            self.size_factor = 10

            if self.subject is not None:
                if os.path.exists(self.data_dir / 'NinaProDB1-subj{}_train.pkl'.format(subject)):
                    with open(self.data_dir / 'NinaProDB1-subj{}_train.pkl'.format(subject), 'rb') as f:
                        self.data = pickle.load(f, encoding='latin1')
                    self.X_aug = self.data['X_augtrain']
                    self.y_aug = self.data['y_augtrain']
                    self.r_aug = self.data['r_augtrain']
                    self.X = self.data['X_train']
                    self.y = self.data['y_train']
                    self.r = self.data['r_train']

                    self._max_len = self.data['max_len_train']
                    self._validate_params() 
                    self._generate(augment=False)

                    if self.shuffle is True:
                        np.random.shuffle(self.indexes)

                else:
                    self._load_dataset()
                    self._validate_params()
                    self._generate()

                    if self.shuffle is True:
                        np.random.shuffle(self.indexes)

                    data = {
                        'X_'+set_: self.X,
                        'y_'+set_: self.y,
                        'r_'+set_: self.r,
                        'X_aug'+set_: self.X_aug,
                        'y_aug'+set_: self.y_aug,
                        'r_aug'+set_: self.r_aug,
                        'max_len_'+set_: self._max_len
                    }

                    with open(self.data_dir / ('NinaProDB1-subj{}_'.format(subject) + set_ + '.pkl'), 'wb') as f:
                        pickle.dump(data, f)
            else:
                if os.path.exists(self.data_dir / 'NinaProDB1-fulldata_train.pkl'):
                    with open(self.data_dir / 'NinaProDB1-fulldata_train.pkl', 'rb') as f:
                        self.data = pickle.load(f, encoding='latin1')
                    self.X_aug = self.data['X_augtrain']
                    self.y_aug = self.data['y_augtrain']
                    self.r_aug = self.data['r_augtrain']
                    self.X = self.data['X_train']
                    self.y = self.data['y_train']
                    self.r = self.data['r_train']

                    self._max_len = self.data['max_len_train']
                    self._validate_params() 
                    self._generate(augment=False)

                    if self.shuffle is True:
                        np.random.shuffle(self.indexes)

                else:
                    self._load_dataset()
                    self._validate_params()
                    self._generate()

                    if self.shuffle is True:
                        np.random.shuffle(self.indexes)

                    data = {
                        'X_'+set_: self.X,
                        'y_'+set_: self.y,
                        'r_'+set_: self.r,
                        'X_aug'+set_: self.X_aug,
                        'y_aug'+set_: self.y_aug,
                        'r_aug'+set_: self.r_aug,
                        'max_len_'+set_: self._max_len
                    }

                    with open(self.data_dir / 'NinaProDB1-fulldata_train.pkl', 'wb') as f:
                        pickle.dump(data, f, protocol=4)
        
        elif set_ == 'test':
            self.batch_size = 1
            self.repetitions = [2, 5, 7]
            self.sample_weight = False
            self.n_reps = len(self.repetitions) 
            self.shuffle = False 
            self.noise_snr_db = 0
            self.time_warping = 0.
            self.mag_warping = 0.
            self.size_factor = 0

            if self.subject is not None: 
                if os.path.exists(self.data_dir / 'NinaProDB1-subj{}_test.pkl'.format(subject)):
                    with open(self.data_dir / 'NinaProDB1-subj{}_test.pkl'.format(subject), 'rb') as f:
                        self.data = pickle.load(f, encoding='latin1')
                    self.X_aug = self.data['X_augtest']
                    self.y_aug = self.data['y_augtest']
                    self.r_aug = self.data['r_augtest']
                    self.X = self.data['X_test']
                    self.y = self.data['y_test']
                    self.r = self.data['r_test']

                    self._max_len = self.data['max_len_test']
                    self._validate_params() 
                    self._generate(augment=False) 

                    if self.shuffle is True:
                        np.random.shuffle(self.indexes)

                else:
                    self._load_dataset()
                    self._validate_params()
                    self._generate()
                    
                    if self.shuffle is True:
                        np.random.shuffle(self.indexes)

                    data = {
                        'X_'+set_: self.X,
                        'y_'+set_: self.y,
                        'r_'+set_: self.r,
                        'X_aug'+set_: self.X_aug,
                        'y_aug'+set_: self.y_aug,
                        'r_aug'+set_: self.r_aug,
                        'max_len_'+set_: self._max_len
                    }
                    
                    with open(self.data_dir / ('NinaProDB1-subj{}_'.format(subject) + set_ + '.pkl'), 'wb') as f:
                        pickle.dump(data, f)
            else:
                if os.path.exists(self.data_dir / 'NinaProDB1-fulldata_test.pkl'):
                    with open(self.data_dir / 'NinaProDB1-fulldata_test.pkl', 'rb') as f:
                        self.data = pickle.load(f, encoding='latin1')
                    self.X_aug = self.data['X_augtest']
                    self.y_aug = self.data['y_augtest']
                    self.r_aug = self.data['r_augtest']
                    self.X = self.data['X_test']
                    self.y = self.data['y_test']
                    self.r = self.data['r_test']

                    self._max_len = self.data['max_len_test']
                    self._validate_params() 
                    self._generate(augment=False)

                    if self.shuffle is True:
                        np.random.shuffle(self.indexes)

                else:
                    self._load_dataset()
                    self._validate_params()
                    self._generate()

                    if self.shuffle is True:
                        np.random.shuffle(self.indexes)

                    data = {
                        'X_'+set_: self.X,
                        'y_'+set_: self.y,
                        'r_'+set_: self.r,
                        'X_aug'+set_: self.X_aug,
                        'y_aug'+set_: self.y_aug,
                        'r_aug'+set_: self.r_aug,
                        'max_len_'+set_: self._max_len
                    }

                    with open(self.data_dir / 'NinaProDB1-fulldata_test.pkl', 'wb') as f:
                        pickle.dump(data, f, protocol=4)
        else:
            raise ValueError("Possible 'set_' values are 'train' or 'test'")

    def __len__(self):
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Generate indexes of the batch
        indexes = self.indexes[idx * self.batch_size:(idx+1) * self.batch_size]

        # Generate data
        output = self._data_generation(indexes)

        #pdb.set_trace()
        sample = {
                'data': output[0].astype(np.float32),
                'target': output[1]
                }

        return sample

    def _data_generation(self, indexes):
        """
        Generates data containing batch_size examples
        """
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)
        if self.sample_weight:
            w = np.empty((self.batch_size), dtype=float)

        # Generate data
        for k, index in enumerate(indexes):
            i, j = self.x_offsets[index]
            # Store sample
            if self.window_size != 0:
                x_aug = np.copy(self.X_aug[i][j:j + self.window_size])
            else:
                x_aug = np.copy(self.X_aug[i])

            if np.prod(x_aug.shape) == np.prod(self.dim):
                x_aug = np.reshape(x_aug, self.dim)
            else:
                raise Exception('Generated sample dimension mismatch. Found {}, expected {}.'.format(x_aug.shape, self.dim))
            
            X[k, ] = x_aug

            # Store class
            y[k] = self.class_index[int(self.y_aug[i])]

            if self.sample_weight:
                w[k] = self.class_weights[(y[k])]
        output = (X, y)

        if self.sample_weight:
            output += (w,)
        return output
    
    def _validate_params(self):
        if ((self.dim[0] is None) or (self.pad_len is None)) and ((self.window_size == 0) or (self.window_step == 0)):
            self.dim = (self._max_len, *self.dim[1:])
            self.pad_len = self._max_len
            self.window_step = 0
            self.window_size = 0

    def _generate(self, augment=True):
        if augment:
            self._augment()
        self._make_segments()
        self.indexes = np.arange(len(self.x_offsets))
        if self.batch_size > len(self.x_offsets):
            self.batch_size = len(self.x_offsets)
        
        self.class_weights = []
        if self.sample_weight:
            self._make_sample_weights()

        if augment:
            if (self.window_size == 0) and (self.pad_len is not None):
                self._pad_sequence(self.X_aug, self.pad_len, self.pad_value) 
                self.X_aug = np.stack(self.X_aug)

    def _augment(self):
        """
        Applies augmentation incrementally
        """
        self.X_aug, self.y_aug, self.r_aug = [], [], []
        for i in range(len(self.X)):
            for _ in range(self.size_factor):
                x = np.copy(self.X[i])
                if self.permutation != 0:
                    x = da.permute(x, nPerm=self.permutation)
                if self.rotation != 0:
                    x = da.rotate(x, rotation=self.rotation, mask=self.rotation_mask)
                if self.time_warping != 0:
                    x = da.time_warp(x, sigma=self.time_warping)
                if self.scale_sigma != 0:
                    x = da.scale(x, sigma=self.scale_sigma)
                if self.mag_warping != 0:
                    x = da.mag_warp(x, sigma=self.mag_warping)
                if self.noise_snr_db != 0:
                    x = da.jitter(x, snr_db=self.noise_snr_db)

                if self.permutation or self.rotation or self.time_warping or self.scale_sigma or self.mag_warping or self.noise_snr_db:
                    self.X_aug.append(x)
                    self.y_aug.append(self.y[i])
                    self.r_aug.append(self.r[i])
            self.X_aug.append(self.X[i])
            self.y_aug.append(self.y[i])
            self.r_aug.append(self.r[i])

    def _make_segments(self):
        """
        Creates segments either using predefined step
        """
        x_offsets = []

        if self.window_size != 0:
            for i in range(len(self.X_aug)):
                for j in range(0, len(self.X_aug[i]) - self.window_size, self.window_step):
                    x_offsets.append((i, j))
        else:
            x_offsets = [(i, 0) for i in range(len(self.X_aug))]

        self.x_offsets = x_offsets

    def _make_sample_weights(self):
        """
        Computes weights for samples
        """
        self.class_weights = np.zeros(self.n_classes)
        for index in self.indexes:
            i, j = self.x_offsets[index]
            self.class_weights[self.class_index[int(self.y_aug[i])]] += 1
        self.class_weights = 1 / self.class_weights
        self.class_weights /= np.max(self.class_weights)

    def _load_dataset(self):
        '''
        Loads data and applies preprocessing function
        '''
        X, y, r = [], [], []
        self._max_len = 0
        if 0 in self.classes:
            # Basically the repetition list is shuffled and then a certain number of classes is choosen and coupled with the
            # corresponding number of repetitions
            rest_rep_groups = list(
                zip(
                    np.random.choice(self.repetitions, (self.n_reps), replace=False),
                    np.random.choice([i for i in self.classes if i != 0], (self.n_reps), replace=False)
                )
            )

        for d in range(len(self.input_directory)):
            for label in [i for i in self.classes if i != 0]:
                for rep in self.repetitions:
                    file = '{}/gesture-{:02d}/{}/rep-{:02d}.mat'.format(self.input_directory[d], int(label), self.data_type, int(rep))
                    data = scipy.io.loadmat(file)
                    x = data['emg'].copy()

                    if self.preprocess_function is not None:
                        if isinstance(self.preprocess_function, list):
                            for params, func in zip(self.preprocess_function_extra, self.preprocess_function):
                                x = func(x, **params)
                        else:
                            x = self.preprocess_function(x, **self.preprocess_function_extra)

                    if len(x) > self._max_len:
                        self._max_len = len(x)
                    X.append(x)
                    y.append(int(np.squeeze(data['stimulus'])[0]))
                    r.append(int(np.squeeze(data['repetition'])[0]))
            
            if 0 in self.classes:
                for rep, label in rest_rep_groups:
                    file = '{}/gesture-00/{}/rep-{:02d}_{:02d}.mat'.format(self.input_directory[d], self.data_type, int(rep), int(label))
                    data = scipy.io.loadmat(file)
                    x = data['emg'].copy()

                    if self.preprocess_function is not None:
                        if isinstance(self.preprocess_function, list):
                            for params, func in zip(self.preprocess_function_extra, self.preprocess_function):
                                x = func(x, **params)
                        else:
                            x = self.preprocess_function(x, **self.preprocess_function_extra)
                    
                    if len(x) > self._max_len:
                        self._max_len = len(x)
                    X.append(x)
                    y.append(int(np.squeeze(data['stimulus'])[0]))
                    r.append(int(np.squeeze(data['repetition'])[0]))
        self.X = X
        self.y = y
        self.r = r

    def _make_class_index(self):
        """
        Maps class label to 0...len(classes)
        """
        #pdb.set_trace()
        self.classes.sort()
        self.class_index = np.zeros(np.max(self.classes) + 1, dtype=int)
        for i, j in enumerate(self.classes):
            self.class_index[j] = i

    def _pad_sequence(self, sequences, max_len, pad_value):
        for i, sequence in enumerate(sequences):
            diff = max_len - sequence.shape[0]
            sequence = np.pad(sequence, ((diff, 0), (0, 0)), 'constant', constant_values=pad_value)
            #pdb.set_trace()
            self.X_aug[i] = sequence

def lpf(x, f=1., fs=100):
    f = f / (fs / 2)
    x = np.abs(x)
    b, a = scipy.signal.butter(1, f, 'low')
    output = scipy.signal.filtfilt(
        b, a, x, axis = 0,
        padtype = 'odd', 
        padlen = 3 * (max(len(b), len(a)) - 1)
    )
    return output

