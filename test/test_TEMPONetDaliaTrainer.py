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
from trainer.TEMPONetDaliaTrainer import TEMPONetDaliaTrainer
from data_loader.DaliaDataLoader import DaliaDataLoader
from model.TEMPONet import TEMPONet
import model.loss as module_loss
import model.metric as module_metric
from utils import prepare_device
from torchinfo import summary

class TestTEMPONetDaliaTrainer(unittest.TestCase):
    
    args_tuple = collections.namedtuple('args', 'config resume device')
    
    def test_cross_val(self):
        # Pars CLI argument and config file
        args = TestTEMPONetDaliaTrainer.args_tuple("config/config_Dalia.json", None, None)
        config = ConfigParser.from_args(args)
        
        # Setup logger
        logger = config.get_logger('train')
        
        # Build model architecture and print to console
        #model = TEMPONet(learned_dil=[1,1,1,1,1,1,1])
        model = TEMPONet('PPG_Dalia', config['arch']['args'],
            learned_dil = [1, 8, 8, 16, 16, 16, 8], 
            learned_rf = [3, 2, 2, 2, 2, 3, 5],
            learned_ch = [8, 8, 64, 32, 32, 128, 64, 96, 128, 256, 128])
        logger.info(model)
        
        mac = (summary(model, (1, 4, 256), verbose=0)).total_mult_adds
        print(f"MAC: {mac} \t FLOPs: {2*mac}")


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
            
            trainer = TEMPONetDaliaTrainer(
                        model = model,
                        criterion = criterion,
                        metric_ftns = metrics,
                        optimizer = optimizer,
                        config = config,
                        device = device,
                        data_loader = DaliaDataLoader,
                        data_dir = config['data_loader']['args']['data_dir'],
                        batch_size = config['data_loader']['args']['batch_size']
                        )

            MAE = trainer.cross_val(config['trainer']['cross_validation']['folds'])
            avg = sum(MAE.values()) / len(MAE)
            print("Average MAE : {}".format(avg))

if __name__ == '__main__':
    unittest.main()
