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
from model.TCResNet14 import TCResNet14 
from torchinfo import summary
import torch
import json
import pdb

class TestTCResNet14(unittest.TestCase):
    
    def test_object_instantiation(self):
        with open('config/config_GoogleSpeechCommands.json', 'r') as f:
            arguments = json.load(f)
        
        model = TCResNet14('GoogleSpeechCommands', arguments['arch']['args'])
    
    def test_feed_data(self):
        with open('config/config_GoogleSpeechCommands.json', 'r') as f:
            arguments = json.load(f)

        model = TCResNet14('GoogleSpeechCommands', arguments['arch']['args'])
        with torch.no_grad():
            data = torch.rand(100, 49, 10)
            output = model(data)
            print(output)
    
    def test_model_architecture(self):
        with open('config/config_GoogleSpeechCommands.json', 'r') as f:
            arguments = json.load(f)
    
        nas_config = arguments['nas']['nas_config']    
        model = TCResNet14('GoogleSpeechCommands', arguments['arch']['args'], do_nas=True, nas_config=nas_config)
        
        mac = (summary(model, (1, 98, 40), verbose=0)).total_mult_adds
        print(2 * mac)

    def test_nas_architecture(self):
        with open('config/config_GoogleSpeechCommands.json', 'r') as f:
            arguments = json.load(f)
    
        nas_config = arguments['nas']['nas_config']    
        model = TCResNet14('GoogleSpeechCommands', arguments['arch']['args'], do_nas=True, nas_config=nas_config)
        summary(model,
                (100, 49, 10),
                verbose = 2,
                col_width = 16,
                col_names=['kernel_size', 'input_size', 'output_size', 'num_params', 'mult_adds']
                )
    
    def test_learned_architecture(self):
        with open('config/config_GoogleSpeechCommands.json', 'r') as f:
            arguments = json.load(f)
        
        nas_config = arguments['nas']['nas_config']
        learned_dil = [1, 1, 4, 1, 8, 8, 128, 256, 256]
        learned_rf = [9, 11, 1, 1, 63, 59, 1, 186, 19]
        learned_ch = [26, 29, 4, 32, 22, 25, 6, 25, 19, 16, 2, 15]
        model = TCResNet14('GoogleSpeechCommands', arguments['arch']['args'], do_nas=False, nas_config=nas_config,
            learned_dil=learned_dil, learned_rf=learned_rf, learned_ch=learned_ch)
        summary(model,
                (1, 98, 40),
                verbose = 2,
                col_width = 16,
                col_names=['kernel_size', 'input_size', 'output_size', 'num_params', 'mult_adds']
                )
        mac = (summary(model, (1, 98, 40), verbose=0)).total_mult_adds
        print(2*mac)
