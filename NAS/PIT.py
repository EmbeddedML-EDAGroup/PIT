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
from utils import prepare_device
import importlib
from pathlib import Path
import model.loss as module_loss
import model.metric as module_metric
from NAS.layer.PITConv1d_v2 import PITConv1d
from NAS.PITTrainer import PITTrainer
from torchinfo import summary
import json
import pdb
import copy

class PIT:
    def __init__(self, config):
        
        self.config = config
        self.target = config['nas']['nas_config']['target']
        self.learn_dil = self.target.get('dilation', False)
        self.learn_rf = self.target.get('filters', False)
        self.learn_ch = self.target.get('channels', False)

        # Configure logger
        self.logger = config.get_logger('train')
        
        # Configure device(s)
        self.device, self.device_ids = prepare_device(config['n_gpu'])
        
        # Configure model
        self.module_arch = importlib.import_module('model.'+config['arch']['type'])
        self.model = config.init_obj('arch', self.module_arch, config['arch']['dataset'], dataset_args=config['arch']['args'], do_nas=config['nas']['do_nas'], nas_config=config['nas']['nas_config'])
        self.logger.info(self.model)
        self.model = self.model.to(self.device)
        if len(self.device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
        # Get loss and metrics
        self.criterion = getattr(module_loss, config['loss'])
        self.metric = [getattr(module_metric, met) for met in self.config['metrics']]

        # Build optimizer 
        self.trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.weight_decay = self.config['optimizer']['args'].get('weight_decay', 0.0)
        self.optimizer = config.init_obj('optimizer', torch.optim, self.trainable_params, self.config['optimizer']['args']['lr'], weight_decay = self.weight_decay)
        #self.optimizer = config.init_obj('optimizer', torch.optim, self.model.parameters())
        self.init_state_optimizer = copy.deepcopy(self.optimizer.state_dict())
        
        # Configure data loader
        self.module_data = importlib.import_module('data_loader.'+config['data_loader']['type'])
        self.data_loader = getattr(self.module_data, config['data_loader']['type'])
        self.data_loader_args = config['data_loader']['args']
        
        self.validation_split = config['data_loader']['args'].get('validation_split', 0.0)
        self.nas_data_loader = self.data_loader(
                data_dir = self.data_loader_args['data_dir'],
                batch_size = self.data_loader_args['batch_size'],
                validation_split = self.validation_split,
		        num_workers = self.data_loader_args['num_workers']
                )
        if self.validation_split != 0.0:
            self.valid_data_loader = self.nas_data_loader.split_validation()
        else:
            self.valid_data_loader = self.data_loader(
                data_dir = self.data_loader_args['data_dir'],
                batch_size = self.data_loader_args['batch_size'],
                set_ = 'valid',
		        num_workers = self.data_loader_args['num_workers']
                )
        # Configure PIT trainer 
        self.pit_trainer = PITTrainer(
                model = self.model,
                criterion = self.criterion,
                metric_ftns = self.metric,
                optimizer = self.optimizer,
                config = self.config,
                device = self.device,
                train_data_loader = self.nas_data_loader,
                valid_data_loader = self.valid_data_loader
                )

        self.learned_arch = dict()

    def warmup(self):
        warmup_epochs = self.config['nas']['nas_config']['warmup_epochs']
        saving_path = Path(self.config['trainer']['save_dir']) / 'warmup_{}.pth'.format(warmup_epochs)
        
        if not saving_path.exists():
            # Freeze NAS weights
            self._freeze_nas_weights(self.pit_trainer.model)
            # Warmup
            self.pit_trainer.warmup(epochs=warmup_epochs)
        # Load pretrained model state dict
        state_dict = torch.load(saving_path)['state_dict']
        state_dict_optimizer = torch.load(saving_path)['optimizer']
        self.pit_trainer.model.load_state_dict(state_dict)
        self.pit_trainer.optimizer.load_state_dict(state_dict_optimizer)
        self._freeze_nas_weights(self.pit_trainer.model)
        if not self.config['nas']['nas_config']['strength']['fixed']:
            print("Train for a single epoch in order to obtain the final loss value")
            self.pit_trainer.warmup(epochs=1)
            self.pit_trainer.model.load_state_dict(state_dict)
            self.pit_trainer.optimizer.load_state_dict(state_dict_optimizer)
        
    def nas(self):
        # Unfreeze NAS weights
        self._freeze_nas_weights(self.pit_trainer.model, unfreeze_dil=self.learn_dil, unfreeze_rf=self.learn_rf, unfreeze_ch=self.learn_ch)
        
        # Search
        self.pit_trainer.nas()

    def search_dil(self, epochs):
        # Unfreeze NAS weights
        self._freeze_nas_weights(self.pit_trainer.model, unfreeze_dil=True, unfreeze_rf=False, unfreeze_ch=False)
        
        # Search
        if epochs == 'max':
            epochs = self.config['trainer']['epochs']
        self.pit_trainer.nas(epochs=epochs, target='gamma')
    
    def search_rf(self, epochs):
        # Unfreeze NAS weights
        self._freeze_nas_weights(self.pit_trainer.model, unfreeze_dil=False, unfreeze_rf=True, unfreeze_ch=False)

        # Search
        if epochs == 'max':
            epochs = self.config['trainer']['epochs']
        self.pit_trainer.nas(epochs=epochs, target='beta')
    
    def search_ch(self, epochs):
        # Unfreeze NAS weights
        self._freeze_nas_weights(self.pit_trainer.model, unfreeze_dil=False, unfreeze_rf=False, unfreeze_ch=True)
        
        # Search
        if epochs == 'max':
            epochs = self.config['trainer']['epochs']
        self.pit_trainer.nas(epochs=epochs, target='alpha')

    def retrain(self):
        # Load learned parameters
        if self.learn_dil:
            path = str(self.pit_trainer.checkpoint_dir / 'learned_arch_dil_last.json')
            with open(path, 'r') as f:
                learned_param_dil = [val for val in json.loads(f.read()).values()]
        if self.learn_rf:
            path = str(self.pit_trainer.checkpoint_dir / 'learned_arch_rf_last.json')
            with open(path, 'r') as f:
                learned_param_rf = [val for val in json.loads(f.read()).values()]
        if self.learn_ch:
            path = str(self.pit_trainer.checkpoint_dir / 'learned_arch_ch_last.json')
            with open(path, 'r') as f:
                learned_param_ch = [val for val in json.loads(f.read()).values()]

        # Build learned model
        nas_config = self.config['nas']['nas_config']
        if (self.learn_dil) and (not self.learn_rf) and (not self.learn_ch):
            learned_model = self.config.init_obj('arch', self.module_arch, self.config['arch']['dataset'], 
                dataset_args=self.config['arch']['args'], nas_config=nas_config, learned_dil=learned_param_dil)
        elif (not self.learn_dil) and (self.learn_rf) and (not self.learn_ch):
            learned_model = self.config.init_obj('arch', self.module_arch, self.config['arch']['dataset'], 
                dataset_args=self.config['arch']['args'], nas_config=nas_config, learned_rf=learned_param_rf)
        elif (not self.learn_dil) and (not self.learn_rf) and (self.learn_ch):
            learned_model = self.config.init_obj('arch', self.module_arch, self.config['arch']['dataset'], 
                dataset_args=self.config['arch']['args'], nas_config=nas_config, learned_ch=learned_param_ch)
        elif (self.learn_dil) and (self.learn_rf) and (not self.learn_ch):
            learned_model = self.config.init_obj('arch', self.module_arch, self.config['arch']['dataset'], 
                dataset_args=self.config['arch']['args'], nas_config=nas_config, learned_dil=learned_param_dil, learned_rf=learned_param_rf)
        elif (not self.learn_dil) and (self.learn_rf) and (self.learn_ch):
            learned_model = self.config.init_obj('arch', self.module_arch, self.config['arch']['dataset'], 
                dataset_args=self.config['arch']['args'], nas_config=nas_config, learned_rf=learned_param_rf, learned_ch=learned_param_ch)
        elif (self.learn_dil) and (not self.learn_rf) and (self.learn_ch):
            learned_model = self.config.init_obj('arch', self.module_arch, self.config['arch']['dataset'], 
                dataset_args=self.config['arch']['args'], nas_config=nas_config, learned_dil=learned_param_dil, learned_ch=learned_param_ch)
        elif (self.learn_dil) and (self.learn_rf) and (self.learn_ch):
            learned_model = self.config.init_obj('arch', self.module_arch, self.config['arch']['dataset'], 
                dataset_args=self.config['arch']['args'], nas_config=nas_config, learned_dil=learned_param_dil, learned_rf=learned_param_rf, learned_ch=learned_param_ch)

        self.logger.info(learned_model)
        learned_model = learned_model.to(self.device)
        if len(self.device_ids) > 1:
            learned_model = torch.nn.DataParallel(learned_model, device_ids=self.device_ids)
        # Setup optimizer
        optimizer = self.config.init_obj('optimizer', torch.optim, learned_model.parameters(), self.config['optimizer']['args']['lr'], weight_decay = self.weight_decay)

        # Retrain
        module_trainer = importlib.import_module('trainer.'+self.config['trainer']['type'])
        retrainer = getattr(module_trainer, self.config['trainer']['type'])
        retrain = retrainer(
                model = learned_model,
                criterion = self.criterion,
                metric_ftns = self.metric,
                optimizer = optimizer,
                config = self.config,
                device = self.device,
                data_loader = self.data_loader,
                data_dir = self.config['data_loader']['args']['data_dir'],
                batch_size = self.config['data_loader']['args']['batch_size']
                )

        if self.config['trainer']['cross_validation']['do']:
            res = retrain.cross_val(self.config['trainer']['cross_validation']['folds'])
            end_res = sum(res.values()) / len(res)
            print("Average res: {}".format(end_res))
            best_res = res
        else:
            retrain.train()
            end_res = retrain.test()
            print("End of training test results: ", end_res)
            best_model = torch.load(retrain.checkpoint_dir / 'model_best.pth')
            retrain.model.load_state_dict(best_model['state_dict'])
            best_res = retrain.test()
            print("Best Model on Validation split test results: ", best_res)
        
        try:
            input_size = torch.rand(torch.tensor(self.nas_data_loader.dataset.X[0]).size())
        except:
            input_size = torch.rand(torch.tensor(self.nas_data_loader.dataset[0]['data'])[1].size())
        self._save_summary(end_res, best_res, learned_model, input_size)

    def learned_model(self):
        raise NotImplementedError

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

    def _save_learned_arch(self):
        path = str(self.pit_trainer.checkpoint_dir / 'learned_arch.json')
        with open(path, 'w') as f:
            json.dump(self.learned_arch, f, indent=4)

    def _save_summary(self, perf_measure_end, perf_measure_best, model, input_size):
        save_dir = self.config.save_dir
        save_path = save_dir / ('summary_'+self.config['name']+'.txt')
        f = open(save_path, 'a+')

        f.write('Regularization Strength: {} \n'.format(self.config['nas']['nas_config']['strength']))
        f.write('Warmup Epochs: {} \n'.format(self.config['nas']['nas_config']['warmup_epochs']))
        f.write('NAS Target: {} \n'.format(self.config['nas']['nas_config']['target']))
        f.write('Mask Type: {} \n'.format(self.config['nas']['nas_config']['mask_type']))
        f.write('Regularizer Target: {} \n'.format(self.config['nas']['nas_config']['regularizer']))
        f.write('Network Size: {} \n'.format(sum(p.numel() for p in model.parameters())))
        try:
            f.write('FLOPs: {} \n'.format(2 * (summary(model, input_size, verbose=0)).total_mult_adds))
        except:
            f.write('FLOPs: {} \n'.format(2 * (summary(model, input_size.unsqueeze(0), verbose=0)).total_mult_adds))
        
        f.write('Performance end of training: {} \n'.format(perf_measure_end))
        f.write('Performance best validation split: {} \n'.format(perf_measure_best))

        f.close()
