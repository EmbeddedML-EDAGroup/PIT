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
import sys
from NAS.PIT import PIT
import pdb

class TestPIT(unittest.TestCase):
    
    args_tuple = collections.namedtuple('args', 'config resume device')
    
    def test_pit_object(self):
        # Pars CLI argument and config file
        args = TestPIT.args_tuple("config/config_ECG5000.json", None, None)
        config = ConfigParser.from_args(args)
        
        # PIT object instantiation
        pit = PIT(config)
        
        # Test Warmup
        pit.warmup()

        # Test Search
        pit.nas()

        # Test Retrain
        pit.retrain()
       
if __name__ == '__main__':
    unittest.main()
