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
from data_loader.DaliaDataLoader import DaliaDataLoader
import test.preprocessing as pp
import numpy as np

class TestDaliaDataset(unittest.TestCase):
    
    def test_object_instantiation(self):
        data_dir = ''
        raise ValueError('Specify data_dir path')
        batch_size = 64
        kfold_it = 0
        set_ = 'train'
        
        data_loader = DaliaDataLoader(
                data_dir=data_dir,
                batch_size=batch_size,
                kfold_it=kfold_it,
                set_ = set_
                )

    def test_object_usage(self):
        data_dir = ''
        raise ValueError('Specify data_dir path')
        batch_size = 128
        kfold_it = 0
        set_ = 'train'
        
        #for set_ in ['train', 'validation', 'test']:
        for set_ in ['validation', 'test']:

            data_loader = DaliaDataLoader(
                    data_dir=data_dir,
                    batch_size=batch_size,
                    kfold_it=kfold_it,
                    set_ = set_
                    )
            
            for batch_idx, batch in enumerate(data_loader):
                print(batch_idx)
                print(batch['data'].size())
                print(batch['target'].size())

                import pdb
                pdb.set_trace()

                if batch_idx == 5:
                    break

if __name__ == '__main__':
    unittest.main()
