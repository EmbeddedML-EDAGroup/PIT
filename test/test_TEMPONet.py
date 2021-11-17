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
from model.TEMPONet import TEMPONet
from torchinfo import summary
import torch
import pdb

from model.loss import MAE

class TestTEMPONet(unittest.TestCase):
    
    def test_object_instantiation(self):
        model = TEMPONet()    

    def test_model_architecture(self):
        model = TEMPONet()
        summ = summary(model, 
                (1, 4, 256), 
                verbose=2,
                col_width = 16,
                col_names=["kernel_size", "input_size", "output_size", "num_params", "mult_adds"]
                )
        print('FLOPs: {}'.format(summ.total_mult_adds * 2))
    
    def test_feed_data(self):
        model = TEMPONet()
        with torch.no_grad():
            data = torch.rand(10, 4, 256)
            output = model(data)
            print(output)
    
    def test_learned_arch(self):
        learned_dil = [1, 1, 1, 1, 1, 1, 1]
        learned_rf = [11, 11, 11, 19, 19, 35, 35]
        learned_ch = [25, 21, 22, 7, 21, 27, 10, 36, 27, 69, 96]
        model = TEMPONet(learned_dil=learned_dil, learned_rf=learned_rf, learned_ch=learned_ch)
        with torch.no_grad():
            data = torch.rand(10, 4, 256)
            output = model(data)
            print(output)
        summ = summary(model, 
                (1, 4, 256), 
                verbose=2,
                col_width = 16,
                col_names=["kernel_size", "input_size", "output_size", "num_params", "mult_adds"]
                )  
        print('FLOPs: {}'.format(summ.total_mult_adds * 2))

if __name__ == '__main__':
    unittest.main()
