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
import numpy as np
from trainer.NinaProDB1Trainer import NinaProDB1Trainer
from data_loader.NinaProDB1DataLoader import NinaProDB1DataLoader
from model.TCCNet import TCCNet
import model.loss as module_loss
import model.metric as module_metric
from utils import prepare_device
import pdb

np.random.seed(1234)

class TestNinaProDB1Trainer(unittest.TestCase):
    
    args_tuple = collections.namedtuple('args', 'config resume device')
    
    def test_cross_val(self):
        # Pars CLI argument and config file
        args = TestNinaProDB1Trainer.args_tuple("config/config_NinaProDB1.json", None, None)
        config = ConfigParser.from_args(args)
        
        # Setup logger
        logger = config.get_logger('train')
        
        # Build model architecture and print to console
        #model = TCCNet('NinaProDB1', config['arch']['args'], learned_dil=[1,1,1,1,1,1,1], learned_rf=[7, 11, 19, 35, 67, 131, 129])
        #model = TCCNet('NinaProDB1', config['arch']['args'], learned_dil=[1,4,8,16,16,32,1])
        model = TCCNet('NinaProDB1', config['arch']['args'])
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
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        
        if config['trainer']['cross_validation']['do']:
            print("Do cross-validation")
            
            trainer = NinaProDB1Trainer(
                        model = model,
                        criterion = criterion,
                        metric_ftns = metrics,
                        optimizer = optimizer,
                        config = config,
                        device = device,
                        data_loader = NinaProDB1DataLoader,
                        data_dir = config['data_loader']['args']['data_dir'],
                        batch_size = config['data_loader']['args']['batch_size']
                        )

            acc = trainer.cross_val(config['trainer']['cross_validation']['folds'])
            avg = sum(acc.values()) / len(acc)
            print("Average Accuracy : {}".format(avg))

if __name__ == '__main__':
    unittest.main()
