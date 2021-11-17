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

import argparse
from parse_config import ConfigParser
import collections
import torch
import torch.multiprocessing as mp
import random
import numpy as np
import sys
import pdb
from NAS.PIT import PIT
import os
import time

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(12345)

def main(config):
    # PIT object instantiation
    pit = PIT(config)

    # Warmup
    if config['nas']['nas_config']['warmup_epochs'] != 0:
        pit.warmup()

    # NAS
    schedule = config['nas']['nas_config']['schedule']
    if (schedule['dilation'] == 'same') and (schedule['filters'] == 'same') and (schedule['channels'] == 'same'):
        t0 = time.time()
        pit.nas()
        print('Elapsed Time: {}'.format(time.time()-t0))
    else:
        for key in schedule.keys():
            print("Search {}".format(key))
            if key == 'dilation':
                pit.search_dil(schedule[key])
            elif key == 'filters':
                pit.search_rf(schedule[key])
            elif key == 'channels':
                pit.search_ch(schedule[key])

    # Retrain
    pit.retrain()

if __name__ == '__main__':

    def is_bool(var):
        if var == 'True':
            return True
        elif var == 'False':
            return False
        else:
            return False
    
    def str_or_int(var):
        if var == 'max' or var == 'same':
            return str(var)
        else:
            return int(var)

    args = argparse.ArgumentParser(description='Config file')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--schedule_dil'], type=str_or_int, target='nas;nas_config;schedule;dilation'),
        CustomArgs(['--schedule_filt'], type=str_or_int, target='nas;nas_config;schedule;filters'),
        CustomArgs(['--schedule_ch'], type=str_or_int, target='nas;nas_config;schedule;channels'),
        CustomArgs(['--dilation', '--search_dilation'], type=is_bool, target='nas;nas_config;target;dilation'),
        CustomArgs(['--filters', '--search_filters'], type=is_bool, target='nas;nas_config;target;filters'),
        CustomArgs(['--channels', '--search_channels'], type=is_bool, target='nas;nas_config;target;channels'),
        CustomArgs(['--mask_dil', '--mask_dilation'], type=str, target='nas;nas_config;mask_type;dilation'),
        CustomArgs(['--mask_filt', '--mask_filters'], type=str, target='nas;nas_config;mask_type;filters'),
        CustomArgs(['--mask_ch', '--mask_channels'], type=str, target='nas;nas_config;mask_type;channels'),
        CustomArgs(['--gr_bin'], type=is_bool, target='nas;nas_config;group_binarize'),
        CustomArgs(['--fix_strength'], type=is_bool, target='nas;nas_config;strength;fixed'),
        CustomArgs(['--strength'], type=float, target='nas;nas_config;strength;value'),
        CustomArgs(['--warmup', '--warmup_epochs'], type=str_or_int, target='nas;nas_config;warmup_epochs')
    ]

    config = ConfigParser.from_args(args, options)
    main(config)
