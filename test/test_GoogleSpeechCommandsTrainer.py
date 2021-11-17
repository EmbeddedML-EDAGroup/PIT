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

import unittest
from parse_config import ConfigParser
import argparse
import collections
import torch
from trainer.GoogleSpeechCommandsTrainer import GoogleSpeechCommandsTrainer
from data_loader.GoogleSpeechCommandsDataLoader import GoogleSpeechCommandsDataLoader
from model.TCResNet14 import TCResNet14
import model.loss as module_loss
import model.metric as module_metric
from utils import prepare_device

import pdb

class TestGoogleSpeechCommandsTrainer(unittest.TestCase):
    
    args_tuple = collections.namedtuple('args', 'config resume device')
    
    def test_training(self):
        # Pars CLI argument and config file
        args = TestGoogleSpeechCommandsTrainer.args_tuple("config/config_GoogleSpeechCommands.json", None, None)
        config = ConfigParser.from_args(args)
        
        # Setup logger
        logger = config.get_logger('train')
        
        # Build model architecture and print to console
        dil = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        rf = [9, 17, 17, 33, 65, 65, 129, 256, 256]
        ch = [32, 35, 19, 31, 21, 37, 10, 20, 14, 20, 1, 21]
        #pdb.set_trace()
        #model = TCResNet14('GoogleSpeechCommands', config['arch']['args'], learned_dil=dil, learned_rf = rf, learned_ch = ch)
        #model = TCResNet14('GoogleSpeechCommands', config['arch']['args'], use_bias=True, pool=True)
        model = TCResNet14('GoogleSpeechCommands', config['arch']['args'], use_bias=True, bn=False, pool=False, learned_dil=dil, learned_rf=rf)#, learned_ch=ch)
        logger.info(model)

        # Prepare for (multi-device) GPU training
        device, device_ids = prepare_device(config['n_gpu'])
        model = model.to(device)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

        # Get function handles of loss and metrics
        criterion = getattr(module_loss, config['loss'])
        metrics = [getattr(module_metric, met) for met in config['metrics']]
        
        # Build optimizer
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        if config['optimizer']['type'] == 'Adam':
            optimizer = config.init_obj('optimizer', torch.optim, trainable_params, 
                config['optimizer']['args']['lr'], weight_decay = config['optimizer']['args']['weight_decay'])
        else:
            optimizer = config.init_obj('optimizer', torch.optim, trainable_params, 
                config['optimizer']['args']['lr'], momentum = 0.9, weight_decay = config['optimizer']['args']['weight_decay'])

            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10000, 0.1)

        trainer = GoogleSpeechCommandsTrainer(
                    model = model,
                    criterion = criterion,
                    metric_ftns = metrics,
                    optimizer = optimizer,
                    config = config,
                    device = device,
                    data_loader = GoogleSpeechCommandsDataLoader,
                    data_dir = config['data_loader']['args']['data_dir'],
                    batch_size = config['data_loader']['args']['batch_size'],
                    do_validation = True
                    )

        trainer.train()
        print(trainer.test())

if __name__ == '__main__':
    unittest.main()
