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
import numpy as np
from numpy import inf
import math
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from NAS.layer.PITConv1d_v2 import PITConv1d
from NAS.regularization.regularizer import Regularizer
import pdb
import json
from torchvision.utils import make_grid
import copy
from pathlib import Path
import pickle
import time

class PITTrainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, train_data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super(PITTrainer, self).__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.train_data_loader = train_data_loader
        if len_epoch is None:
            # Epoch-based training
            self.len_epoch = len(self.train_data_loader)
        else:
            # Iteration-based training
            self.train_data_loader = inf_loop(train_data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.num_workers = config['data_loader']['args']['num_workers']
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(self.config['data_loader']['args']['batch_size']))

        self.do_warmup = False
        self.do_nas = False
        self.fix_strength = config['nas']['nas_config']['strength']['fixed']
        if self.fix_strength:
            self.strength = config['nas']['nas_config']['strength']['value']
        else:
            self.reg_fraction = config['nas']['nas_config']['strength']['value']
            self.loss_final = 0.
        
        self.target = config['nas']['nas_config']['target']
        self.learn_dil = self.target.get('dilation', False)
        self.learn_rf = self.target.get('filters', False)
        self.learn_ch = self.target.get('channels', False)

        self.mask_type = config['nas']['nas_config']['mask_type']
        self.mask_op = self.mask_type.get('type', 'sum')
        self.mask_dil = self.mask_type.get('dilation', 'binary')
        self.mask_rf = self.mask_type.get('filters', 'binary')
        self.mask_ch = self.mask_type.get('channels', 'binary')

        # Define and initialize temperature
        if self.mask_ch == 'fbnet':
            self.temp = config['nas']['nas_config']['tau']
                
        self.regularizer = self.config['nas']['nas_config']['regularizer']
        try:
            input_sample = torch.rand(torch.tensor(self.train_data_loader.dataset.X)[0].size())
        except:
            input_sample = torch.rand(torch.tensor(self.train_data_loader.dataset[0]['data'])[1].size())
        self.output_shapes = self._eval_output_shape(input_sample)

        self.PIT_layers = list(filter(lambda x: (isinstance(x, PITConv1d)), self.model.modules()))

        self.train_metrics = MetricTracker('loss', 'reg_loss', 'acc_loss',*[m.__name__ for m in self.metric_ftns], writer=self.writer)
        if self.do_validation:
            self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        self.learned_arch_dil = dict()
        self.learned_arch_rf = dict()
        self.learned_arch_ch = dict()
        
    def warmup(self, epochs=None):
        self.do_warmup = True
        self.epochs = self.config['trainer']['epochs'] if epochs=='max' else epochs
        self.train()
        self.do_warmup = False

    def nas(self, epochs=None, target=None):
        self.do_nas = True
        self.reg_target = target
        self.epochs = self.config['trainer']['epochs'] if epochs is None else epochs
        if not self.fix_strength and target is not None:
            self.strength = self._update_strength_value(self.reg_fraction, self.loss_final)
            print("Previous Loss value: {}, fraction: {}, Strength: {}".format(self.loss_final, self.reg_fraction, self.strength))
        elif not self.fix_strength and target is not None:
            raise ValueError("Non-fixed strength values make sense only if the search si performed on seprated target separately")
        self.train()
        self.do_nas = False

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        old_loss = 0
        epoch = 0

        for epoch in range(self.start_epoch, self.epochs+self.start_epoch):
            if self.mask_ch != 'fbnet':
                # Normal Training
                result = self._train_epoch(epoch, self.train_data_loader)
            else:
                # Split dataset in 80% and 20%
                data = self.train_data_loader.dataset
                len_data_nw = int(len(data) * 0.8) # 80%, normal-weights
                len_data_np = len(data) - len_data_nw # 20%, nas-params
                data_nw, data_np = torch.utils.data.random_split(data, [len_data_nw, len_data_np])
                nw_loader = torch.utils.data.DataLoader(
                    data_nw, batch_size=self.config['data_loader']['args']['batch_size'], 
                    shuffle=True, num_workers=self.config['data_loader']['args']['num_workers'],
                    pin_memory=True
                    ) 
                np_loader = torch.utils.data.DataLoader(
                    data_np, batch_size=self.config['data_loader']['args']['batch_size'],
                    shuffle=True, num_workers=self.config['data_loader']['args']['num_workers'],
                    pin_memory=True
                    )
                # Freeze NAS parameters and train normal weights with 80%-split
                self._freeze_nas_weights(self.model, unfreeze_ch=False)
                result = self._train_epoch(epoch, nw_loader)
                # Unfreeze NAS parameters
                self._freeze_nas_weights(self.model, unfreeze_ch=True)
                # Freeze normal weights and train NAS parameters with 20%-split
                self._freeze_normal_weights(self.model, unfreeze=False)
                result = self._train_epoch(epoch, np_loader)
                # Unfreeze normal weights
                self._freeze_normal_weights(self.model, unfreeze=True)
            
            ## Anneal Temperature ##
            # Compute new Temp
            if self.mask_ch == 'fbnet':
                self.temp = self.temp * math.exp(-0.045)
                # Update Temp in every PIT Layer
                for layer in self.PIT_layers:
                    layer.tau = self.temp

            # save logged informations into log dicts
            log = {'epoch' : epoch}
            log.update(result)

            # print logged information to the screen
            for key, value in log.items():
                self.logger.info('{:15s}: {}'.format(str(key), value))
            
            # Update regularization strength
            old_loss = log['loss']

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether the model performance improved or not, according to specified metric (mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best - 0.0001) or (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best + 0.00001)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found." "Model performance monitoring is disabled".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False
            
            if improved and epoch >= 10:
                self.mnt_best = log[self.mnt_metric]
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1
                self.logger.info("Validation performance didn\'t improve for {} epochs.".format(not_improved_count))
                self.logger.info("Keep searching for {} epochs.".format(self.early_stop - not_improved_count))

            if not_improved_count > self.early_stop:
                self.logger.info("Validation performance didn\'t improve for {} epochs." "Training Stops.".format(self.early_stop))
                self._save_checkpoint(epoch, save_last=True)
                break

            if epoch % self.save_period == 0 or best==True:
                if epoch >= 5:
                    self._save_checkpoint(epoch, save_best=best)
        if not self.fix_strength:
            self.loss_final = old_loss 
         
        self._save_checkpoint(epoch, save_last=True)
        
    def _save_checkpoint(self, epoch, save_best=False, save_last=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
            }
        if not self.do_warmup:
            if save_best:
                best_path = str(self.checkpoint_dir / 'model_best.pth')
                torch.save(state, best_path)
                self.logger.info("Saving current best: model_best.pth ...")
                self.logger.info("Extracting learned architecture: learned_arch_best.json ...")
                self._extract_learned_arch()
                self._save_learned_arch('best')
            elif save_last:
                last_path = str(self.checkpoint_dir / 'model_last.pth')
                torch.save(state, last_path)
                self.logger.info("Saving current best: model_last.pth ...")
                self.logger.info("Extracting learned architecture: learned_arch_last.json ...")
                self._extract_learned_arch()
                self._save_learned_arch('last')
            else:
                filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
                torch.save(state, filename)
                self.logger.info("Saving checkpoint: {} ...".format(filename))
                self.logger.info("Extracting learned architecture: learned_arch_checkpoint-epoch{}.json ...".format(epoch))
                self._extract_learned_arch()
                self._save_learned_arch('checkpoint-epoch{}'.format((epoch)))
        else:
            if save_best or save_last:
                warmup_path = Path(self.config['trainer']['save_dir']) / 'warmup_{}.pth'.format(self.config['nas']['nas_config']['warmup_epochs'])
                torch.save(state, warmup_path)
                self.logger.info("Saving warmup output: {}...".format(warmup_path))

    def _extract_learned_arch(self):
        if self.learn_dil:
            for name, child in self.model.named_modules():
                if isinstance(child, PITConv1d):
                    if not child.fc and child.stride == 1:
                        if self.mask_dil == 'stochastic':
                            self.learned_arch_dil[name] = self._dilation_factor(child.bin_gamma, child.group_bin)
                        else:
                            self.learned_arch_dil[name] = self._dilation_factor(child.gamma, child.group_bin)
        if self.learn_rf:
            for name, child in self.model.named_modules():
                if isinstance(child, PITConv1d):
                    if not child.fc and child.stride == 1:
                        if self.mask_rf == 'stochastic':
                            self.learned_arch_rf[name] = self._filter_size(child.bin_beta, child.granularity, child.group_bin)
                        else:
                            self.learned_arch_rf[name] = self._filter_size(child.beta, child.granularity, child.group_bin)
        if self.learn_ch:
            for name, child in self.model.named_modules():
                if isinstance(child, PITConv1d) and child.track_arch:
                    if self.mask_ch == 'stochastic':
                        self.learned_arch_ch[name] = self._channel_number(child.bin_alpha)
                    elif self.mask_ch == 'fbnet':
                        self.learned_arch_ch[name] = self._channel_number_fb(child.fbnet_mask, 
                            child.bin_alpha)
                    else:
                        self.learned_arch_ch[name] = self._channel_number(child.alpha, child.is_residual)
    
    def _save_learned_arch(self, name):
        if self.learn_dil:
            path = str(self.checkpoint_dir / ('learned_arch_dil_'+name+'.json'))
            with open(path, 'w') as f:
                json.dump(self.learned_arch_dil, f, indent=4)
        if self.learn_rf:
            path = str(self.checkpoint_dir / ('learned_arch_rf_'+name+'.json'))
            with open(path, 'w') as f:
                json.dump(self.learned_arch_rf, f, indent=4)
        if self.learn_ch:
            path = str(self.checkpoint_dir / ('learned_arch_ch_'+name+'.json'))
            with open(path, 'w') as f:
                json.dump(self.learned_arch_ch, f, indent=4)

    def _train_epoch(self, epoch, data_loader):
        """
        Training logic for an epoch
                   
        :param epoch: Integer, current training epoch.
        :return : A log that contains average loss and metric in this epoch
        """
        self.model.train()
        self.train_metrics.reset()
        t0 = time.time() 
        for batch_idx, batch in enumerate(data_loader):
            data, target = batch['data'].to(self.device), batch['target'].to(self.device)

            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.criterion(output, target)
            
            regularization_loss = 0
            if self.do_nas and not self.do_warmup:
                regularizer = self.config['nas']['nas_config']['regularizer']
                reg_loss = 0
                layer_0 = None
                layer_1 = None
                PIT_layers = list(filter(lambda x: (isinstance(x, PITConv1d)), self.model.modules()))
                for idx, layer in enumerate(PIT_layers):
                    if layer.prev_layer == 'Input':
                        layer_0 = layer
                    else:
                        layer_0 = list(filter(lambda x: (x.name == layer.prev_layer), PIT_layers))[0]
                    layer_1 = layer 
                    actual_shape = self.output_shapes[idx]
                    reg_loss += self.strength * self._compute_reg_loss(layer_0, layer_1, actual_shape, self.target, self.regularizer)
                regularization_loss = reg_loss
                loss += regularization_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('reg_loss', regularization_loss)
            self.train_metrics.update('acc_loss', loss.item() - regularization_loss)

            for metr in self.metric_ftns:
                self.train_metrics.update(metr.__name__, metr(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(epoch, self._progress(batch_idx), loss.item()))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()
        
        # Eval elapsed time
        print('Epoch elapsed Time: {}'.format(time.time()-t0))

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

        return self.valid_metrics.result()
    
    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_data_loader, 'n_samples'):
            current = batch_idx #* self.config['data_loader']['args']['batch_size']
            total = self.train_data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _compute_reg_loss(self, prev_layer, layer, shape, nas_target, metric):
        mask_type = self.config['nas']['nas_config']['mask_type']
        reg = Regularizer(prev_layer, layer, mask_type, metric, nas_target, self.reg_target, shape, self.device, nas_config=self.config['nas']['nas_config'])
        return reg.reg_loss
    
    def _dilation_factor(self, gamma, group_gamma):
        with torch.no_grad(): 
            if group_gamma:
                if self.mask_op == 'mul':
                    gamma_bin = torch.tensor([torch.prod(gamma[:i]) for i in reversed(range(1, gamma.size()[0]+1))])
                    gamma_bin = gamma_bin > 0.5
                    gamma_bin = gamma_bin.type(torch.int) 
                    gamma_list = gamma_bin.tolist()
                elif self.mask_op == 'sum':
                    gamma_bin = torch.tensor([torch.sum(gamma[:i]) for i in range(1, gamma.size()[0]+1)])
                    gamma_bin = gamma_bin > 0.5
                    gamma_bin = gamma_bin.type(torch.int) 
                    gamma_list = gamma_bin.tolist()
            else:
                gamma_bin = gamma > 0.5
                gamma_bin = gamma_bin.type(torch.int) 
                gamma_list = gamma_bin.tolist()
                gamma_list.reverse()
        dil = 0
        for i in gamma_list:
            if i <= 0:
                dil += 1
            else:
                break
        return 2 ** dil
    
    def _filter_size(self, beta, granularity, group_beta):
        with torch.no_grad():
            if group_beta:
                if self.mask_op == 'mul':
                    beta_bin = torch.tensor([torch.prod(beta[:i]) for i in reversed(range(1, beta.size()[0]+1))])
                    beta_bin = beta_bin > 0.5
                    beta_bin = beta_bin.type(torch.int) 
                    beta_list = beta_bin.tolist()
                elif self.mask_op == 'sum':
                    beta_bin = torch.tensor([torch.sum(beta[:i]) for i in range(1, beta.size()[0]+1)])
                    beta_bin = beta_bin > 0.5
                    beta_bin = beta_bin.type(torch.int) 
                    beta_list = beta_bin.tolist()
            else:
                beta_bin = beta > 0.5
                beta_bin = beta_bin.type(torch.int) 
                beta_list = beta_bin.tolist()
                beta_list.reverse()
        kernel_size = 1
        for i in reversed(beta_list):
            if i <= 0:
                break
            else:
                kernel_size += granularity
        return kernel_size

    def _channel_number(self, alpha, is_residual=False):
        mask_ch = self.config['nas']['nas_config']['mask_type']['channels']
        with torch.no_grad():
            if mask_ch == 'binary' or mask_ch == 'linear':
                out_channels = int((alpha > 0.5).sum())
                if (out_channels == 0) and (not is_residual):
                    out_channels = 1
                elif not is_residual:
                    out_channels += 1
            elif mask_ch == 'stochastic':
                #out_channels = int(torch.sigmoid(alpha).sum())
                out_channels = int((alpha > 0.5).sum())
                if out_channels == 0:
                    out_channels = 1
        
        return out_channels

    def _channel_number_fb(self, mask, alpha):
        with torch.no_grad():
            # Compute mask-idx with highest associated probability
            mask_idx = torch.argmax(alpha)
            # Compute number of channels for chosen mask
            out_channels = int(sum(mask[mask_idx]))
        return out_channels


    def _update_strength_value(self, fraction, loss):
        return fraction * loss / self._eval_initial_reg_loss(self.model)
    
    def _eval_initial_reg_loss(self, model):
        layer_0 = None
        layer_1 = None
        reg_loss = 0.
        no_prev = True
        nas_target = self.config['nas']['nas_config']['target']
        regularizer = self.config['nas']['nas_config']['regularizer']
        with torch.no_grad():
            for name, child in self.model.named_modules():
                if isinstance(child, PITConv1d):
                    if no_prev:
                        layer_0 = child
                        layer_1 = child
                        no_prev = False
                    else:
                        layer_1 = child
                    mask_type = self.config['nas']['nas_config']['mask_type']
                    reg = Regularizer(layer_0, layer_1, mask_type, regularizer, nas_target, self.reg_target, nas_config=self.config['nas']['nas_config'])
                    reg_loss += reg.reg_loss
            return reg_loss

    def _eval_output_shape(self, input_sample):
        if self.device.type == 'cuda':
            input_sample = input_sample.cuda()
        outputs = list()
        PIT_layers = list(filter(lambda x: (isinstance(x, PITConv1d)), self.model.modules()))
        actual_shape = None

        def _local_hook(_module, _input, _output):
            nonlocal actual_shape
            actual_shape = _output.size()
            return _output

        self.model.eval()
        with torch.no_grad():
            for layer in PIT_layers:
                layer.register_forward_hook(_local_hook)
                self.model(input_sample.unsqueeze(0))
                outputs.append(actual_shape)
        return outputs

    def _freeze_nas_weights(self, model, unfreeze_dil=False, unfreeze_rf=False, unfreeze_ch=False):
        for _, child in model.named_modules():
            if isinstance(child, PITConv1d):
                for name, param in child.named_parameters():
                    if name != 'weight' and name != 'bias':
                        if name == 'gamma':
                            param.requires_grad = unfreeze_dil
                        elif name == 'beta':
                            param.requires_grad = unfreeze_rf
                        elif name == 'alpha':
                            param.requires_grad = unfreeze_ch

    def _freeze_normal_weights(self, model, unfreeze=False):
        for name, param in model.named_parameters():
            if not 'gamma' in name and not 'beta' in name and not 'alpha' in name:
                param.requires_grad = unfreeze

    def _strength_schedule(self, epoch, actual_loss, old_loss):
        init_strength = self.strength
        if epoch < 0:
            loss_diff = old_loss - actual_loss
            frac = loss_diff / old_loss
            self.strength = init_strength * (1 + frac + epoch/100)
            if epoch % 5 == 0:
                print("Old Strength: {} New Strength:{}".format(init_strength, self.strength))

    def _plot_NAS_weight(self):
        for module_name, module in self.model.named_modules():
            if isinstance(module, PITConv1d):
                for name, param in module.named_parameters():
                    if name != 'weight' and name != 'bias':
                        i = 0
                        for element in param:
                            self.writer.add_scalar(name+str(i)+'/'+module_name, element)
                            i += 1
    
    def _plot_NAS_weight_bin(self):
        for module_name, module in self.model.named_modules():
            if isinstance(module, PITConv1d):
                for name, param in module.named_parameters():
                    if name == 'gamma' or name == 'beta' or name == 'alpha':
                        i = 0
                        bin_param = getattr(module, 'bin_'+name)
                        for element in bin_param:
                            self.writer.add_scalar('bin_'+name+str(i)+'/'+module_name, element)
                            i += 1

    def _plot_NAS_weight_grad(self):
        for module_name, module in self.model.named_modules():
            if isinstance(module, PITConv1d):
                for name, param in module.named_parameters():
                    if name != 'weight' and name != 'bias':
                        i = 0
                        try:
                            for grad in param.grad:
                                self.writer.add_scalar('grad_'+name+str(i)+'/'+module_name, grad)
                                i += 1
                        except:
                            pass
    
    
