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
from trainer.ECG5000Trainer import ECG5000Trainer
from data_loader.ECG5000DataLoader import ECG5000DataLoader
from model.ECGTCN import ECGTCN
import model.loss as module_loss
import model.metric as module_metric
from utils import prepare_device

class TestECG5000Trainer(unittest.TestCase):
    
    args_tuple = collections.namedtuple('args', 'config resume device')
    
    def test_training(self):
        # Pars CLI argument and config file
        args = TestECG5000Trainer.args_tuple("config/config_ECG5000.json", None, None)
        config = ConfigParser.from_args(args)
        
        # Setup logger
        logger = config.get_logger('train')
        
        # Build model architecture and print to console
        #dil = [1, 1, 1, 1, 1, 1]
        #rf = [11, 11, 21, 21, 41, 41]
        #model = ECGTCN('ECG5000', config['arch']['args'], learned_dil = dil, learned_rf = rf)
        model = ECGTCN('ECG5000', config['arch']['args'])
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

        trainer = ECG5000Trainer(
                    model = model,
                    criterion = criterion,
                    metric_ftns = metrics,
                    optimizer = optimizer,
                    config = config,
                    device = device,
                    data_loader = ECG5000DataLoader,
                    data_dir = config['data_loader']['args']['data_dir'],
                    batch_size = config['data_loader']['args']['batch_size'],
                    do_validation = True
                    )

        if config['trainer']['cross_validation']['do']:
            res = trainer.cross_val(config['trainer']['cross_validation']['folds'])
            end_res = sum(res.values()) / len(res)
            print("Average res: {}".format(end_res))
            best_res = res
        else:
            trainer.train()
        print(trainer.test())

if __name__ == '__main__':
    unittest.main()
