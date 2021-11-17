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

import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import copy

class TEMPONetDaliaTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, data_loader, data_dir, batch_size, do_validation=True, lr_scheduler=None, len_epoch=None):
        super(TEMPONetDaliaTrainer, self).__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.len_epoch = len_epoch
        self.do_validation = do_validation
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.test_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def cross_val(self, n_folds):
        MAE = dict()
        init_state_model = copy.deepcopy(self.model.state_dict())
        init_state_optimizer = copy.deepcopy(self.optimizer.state_dict())
        for i in range(n_folds):
            print('Iteration {}/{}'.format(i+1, n_folds))
            # Reload initial model state
            self.model.load_state_dict(init_state_model)
            
            # Reload initial optimizer state
            self.optimizer.load_state_dict(init_state_optimizer)

            # Build data loaders for the current fold
            self.train_data_loader = self.data_loader(
                    data_dir = self.data_dir,
                    batch_size = self.batch_size,
                    kfold_it = i,
                    set_ = 'train'
                    )
            self.valid_data_loader = self.data_loader(
                    data_dir = self.data_dir,
                    batch_size = self.batch_size,
                    kfold_it = i,
                    set_ = 'validation'
                    )
            self.test_data_loader = self.data_loader(
                    data_dir = self.data_dir,
                    batch_size = self.batch_size,
                    kfold_it = i,
                    set_ = 'test'
                    )
            
            self.len_epoch = len(self.train_data_loader)
            
            self.train()
            
            subj = self.test_data_loader.dataset.get_test_subj()
            res = self.test()
            MAE[subj] = res['MAE'].cpu()
            print("Subj {} : {}".format(subj, MAE[subj]))
            print("MAE : {}".format(MAE))
        return MAE
    
    def test(self):
        self.model.eval()
        self.test_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_data_loader):
                data, target = batch['data'].to(self.device), batch['target'].to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                self.test_metrics.update('loss', loss.item())
                for metr in self.metric_ftns:
                    self.test_metrics.update(metr.__name__, metr(output, target))
        return self.test_metrics.result()


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return : A log that contains average loss and metric in this epoch
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, batch in enumerate(self.train_data_loader):
            data, target = batch['data'].to(self.device), batch['target'].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1)* self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for metr in self.metric_ftns:
                self.train_metrics.update(metr.__name__, metr(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(epoch, self._progress(batch_idx), loss.item()))
            
            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation.
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_data_loader):
                data, target = batch['data'].to(self.device), batch['target'].to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for metr in self.metric_ftns:
                    self.valid_metrics.update(metr.__name__, metr(output, target))

        # Add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_data_loader, 'n_samples'):
            current = batch_idx * self.train_data_loader.batch_size
            total = self.train_data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
