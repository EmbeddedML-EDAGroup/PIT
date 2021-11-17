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
from model.TCCNet import TCCNet 
from torchinfo import summary
import torch
import json
import pdb

class TestTCCNet(unittest.TestCase):
    
    def test_object_instantiation(self):
        with open('config/config_NinaProDB1.json', 'r') as f:
            arguments = json.load(f)
        
        #learned_dil = [2, 1, 2, 16, 32, 64]
        #model = TCCNet('NinaProDB1', arguments['arch']['args'], learned_dil=learned_dil)
        model = TCCNet('NinaProDB1', arguments['arch']['args'])

    def test_plain_architecture(self):
        with open('config_NinaProDB1.json', 'r') as f:
            arguments = json.load(f)
        model = TCCNet('NinaProDB1', arguments['arch']['args'], do_nas=False)
        summ = summary(model,
                (30, 150, 10),
                verbose = 2,
                col_width = 16,
                col_names=['kernel_size', 'input_size', 'output_size', 'num_params', 'mult_adds']
                )
        print('FLOPs: {}'.format(summ.total_mult_adds * 2))


    def test_learned_architecture(self):
        with open('config/config_NinaProDB1.json', 'r') as f:
            arguments = json.load(f)
        nas_config = arguments['nas']['nas_config'] 
        learned_dil = [1, 1, 2, 2, 2, 2, 4]
        learned_rf = [3, 3, 7, 13, 15, 63, 45]
        learned_ch = [32, 32, 10, 31, 12, 41, 82]
        model = TCCNet('NinaProDB1', arguments['arch']['args'], do_nas=False, nas_config=nas_config,
            learned_dil=learned_dil, learned_rf=learned_rf, learned_ch=learned_ch)
        summ = summary(model,
                (30, 150, 10),
                verbose = 2,
                col_width = 16,
                col_names=['kernel_size', 'input_size', 'output_size', 'num_params', 'mult_adds']
                )
        print('FLOPs: {}'.format(summ.total_mult_adds * 2))
    
