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
from model.ECGTCN import ECGTCN
from torchinfo import summary
import torch
import json
import pdb

class TestECGTCN(unittest.TestCase):
    
    def test_object_instantiation(self):
        with open('config/config_ECG5000.json', 'r') as f:
            arguments = json.load(f)
        
        model = ECGTCN(arguments['arch']['args'])
    
    def test_model_architecture(self):
        with open('config/config_ECG5000.json', 'r') as f:
            arguments = json.load(f)
        
        model = ECGTCN(arguments['arch']['args'], do_nas=False, nas_config={'target': 'dilation', 'mask_type': 'stochastic', 'regularizer': 'size', 'strength': 5e-6, 'warmup_epochs': 5})
        summary(model,
                (30, 140),
                verbose = 2,
                col_width = 16,
                col_names=['kernel_size', 'input_size', 'output_size', 'num_params', 'mult_adds']
                )
    
    def test_learned_arch(self):
        with open('config/config_ECG5000.json', 'r') as f:
            arguments = json.load(f)
            
        learned_dil = [1, 1, 1, 1, 1, 1]
        learned_rf = [11, 11, 21, 21, 41, 41]
        learned_ch = [11, 11, 11, 11, 11, 11]
        model = ECGTCN(dataset_args=arguments['arch']['args'], learned_dil=learned_dil, learned_rf=learned_rf, learned_ch=learned_ch)
        with torch.no_grad():
            data = torch.rand(30, 140)
            output = model(data)
            print(output)
        summ = summary(model, 
                (40, 140), 
                verbose=2,
                col_width = 16,
                col_names=["kernel_size", "input_size", "output_size", "num_params", "mult_adds"]
                )  
        print('FLOPs: {}'.format(summ.total_mult_adds * 2))
