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
from data_loader.ECG5000DataLoader import ECG5000DataLoader
import test.preprocessing as pp
import numpy as np
import pdb

class TestECG5000DataLoader(unittest.TestCase):
    
    def test_trainset(self):
        data_dir = ''
        raise ValueError('Specify data_dir path')
        batch_size = 30
        set_ = ['train', 'test']
        
        for s in set_:
            data_loader = ECG5000DataLoader(
                    data_dir=data_dir,
                    batch_size=batch_size,
                    shuffle=True,
                    set_ = s
                    )
    
            for batch_idx, batch in enumerate(data_loader):
                print('Dataset: {}, split: {}'.format(data_dir, s))
                print(batch_idx)
                print(batch['data'].size())
                print(batch['target'].size())

                if batch_idx == 2:
                    break

if __name__ == '__main__':
    unittest.main()
