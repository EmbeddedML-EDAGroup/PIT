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
from NAS.layer.PITConv1d_v2 import PITConv1d
import torch
import pdb

nas_config = {
    "schedule": {
        "channels": "same",
        "filters": "same",
        "dilation": "same"
        },
    "target": {
        "dilation":True,
        "filters": True,
        "channels": True
        },
    "mask_type": {
        "type": "mul",
        "dilation": "binary",
        "filters": "binary",
        "channels": "binary"
        },
    "hysteresis": {
        "do": True,
        "eps": 0.0
        },
    "regularizer": "size",
    "rf_expansion_ratio": 2,
    "granularity": 2,
    "tau": 0.001,
    "strength":{
        "fixed": True,
        "value": 5e-8
        },
    "warmup_epochs": 5
    }

class TestPITConv1d(unittest.TestCase):
        
    def test_size_parameters(self):
        layer = PITConv1d(32, 64, 11, NAS_features=nas_config)
        
        print("\n")
        print("Weights shape : {}".format(layer.weight.size()))
        
        print("Gamma shape : {}".format(layer.gamma.size()))
        print("Beta shape : {}".format(layer.beta.size()))
        print("Alpha shape : {}".format(layer.alpha.size()))

    def test_mask(self):
       layer = PITConv1d(32, 64, 11, NAS_features=nas_config)
       mask = layer._dilation_mask_sum(layer.weight, layer.gamma)

    def test_feed_data(self):
        layer = PITConv1d(32, 64, 11, NAS_features=nas_config)
        
        x = torch.randn((32,32,256))
        out = layer(x)
        #pdb.set_trace()

if __name__ == '__main__':
    unittest.main()
