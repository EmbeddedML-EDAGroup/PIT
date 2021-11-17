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

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from base import BaseModel
from NAS.layer.PITConv1d_v2 import PITConv1d
from math import ceil
import pdb

class ECGTCN(BaseModel):
    """
    ECGTCN architecture:
    Variable number of repeated instances of TemporalBlock.
    """
    def __init__(self, dataset_name='ECG5000', dataset_args={}, do_nas=False, nas_config={}, learned_dil=[], learned_rf=[], learned_ch=[]):
        super(ECGTCN, self).__init__()
        
        self.do_nas = do_nas
        self.nas_config = nas_config
        self.dataset_name = dataset_name
        self.dataset_args = dataset_args
        self.num_channels = [self.dataset_args['nhid']] * self.dataset_args['levels']
        
        self.pad0 = nn.ConstantPad1d(
                padding = (self.dataset_args['kernel_size']-1, 0),
                value = 0
                )
        self.conv0 = nn.Conv1d(
                in_channels = self.dataset_args['input_size'],
                out_channels = self.dataset_args['input_size']+1,
                kernel_size = self.dataset_args['kernel_size'],
                bias = True
                )
        self.act0 = nn.ReLU()
        self.batchnorm0 = nn.BatchNorm1d(
                num_features = self.dataset_args['input_size']+1,
                )
        
        self.tcn = TempNet(
                num_inputs = self.dataset_args['input_size']+1, 
                num_channels = self.num_channels, 
                kernel_size = self.dataset_args['kernel_size'], 
                dropout = self.dataset_args['dropout'], 
                learned_dil = learned_dil, 
                learned_rf = learned_rf, 
                learned_ch = learned_ch,
                do_nas = do_nas, 
                nas_config = nas_config
                )
        if not learned_ch: 
            self.out = nn.Linear(
                    in_features = self.num_channels[-1]*140,
                    out_features = self.dataset_args['output_size'],
                    bias = True
                    )
        else:
            self.out = nn.Linear(
                    in_features = learned_ch[-1]*140,
                    out_features = self.dataset_args['output_size'],
                    bias = True
                    )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pad0(x)
        x = self.conv0(x)
        x = self.batchnorm0(x)
        x = self.act0(x)
        y1 = self.tcn(x)
        out = y1.flatten(1)
        out = self.out(out)
        out = F.log_softmax(out, dim=1)
        return out

class TempNet(BaseModel):
    """
    Temporal Convolutional Net composed of a number of TempBlock defined with a specific parameter.
    """
    def __init__(self, num_inputs, num_channels, kernel_size, dropout=0.2, learned_dil=[], learned_rf=[], learned_ch=[], do_nas=False, nas_config={}):
        super(TempNet, self).__init__()
        layers = list()
        num_levels = len(num_channels)
        original_rf = [(kernel_size - 1) * (2**i) + 1 for i in range(num_levels)]
        for i in range(num_levels):
            dilation_size = list()
            k = list()
            if not learned_dil:
                dilation_size.append(2 ** i)
                dilation_size.append(2 ** i)
            else:
                dilation_size.append(learned_dil[2*i])
                dilation_size.append(learned_dil[2*i+1])
            if not learned_rf:
                k.append(ceil(original_rf[i]/dilation_size[0]))
                k.append(ceil(original_rf[i]/dilation_size[1]))
            elif not learned_dil:
                dilation_size[0] = 1
                dilation_size[1] = 1
                k.append(ceil(learned_rf[2*i]/dilation_size[0]))
                k.append(ceil(learned_rf[2*i+1]/dilation_size[1]))
            else:
                k.append(ceil(learned_rf[2*i]/dilation_size[0]))
                k.append(ceil(learned_rf[2*i+1]/dilation_size[1]))
            if not learned_ch:
                in_channels = [num_inputs, num_channels[i-1]] if i == 0 else [num_channels[i-1], num_channels[i-1]]
                out_channels = [num_channels[i], num_channels[i]]
            else:
                in_channels = [num_inputs, learned_ch[i]] if i == 0 else [learned_ch[2*i-1], learned_ch[2*i]]
                out_channels = [learned_ch[2*i], learned_ch[2*i+1]]
            layers += [TempBlock(
                ch_in = in_channels, 
                ch_out = out_channels, 
                k_size = k, 
                dil = dilation_size,
                dropout = dropout,
                block_idx = i,
                do_nas = do_nas,
                nas_config = nas_config
                )]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TempBlock(BaseModel):
    """
    Temporal Block composed of two temporal convolutional block.
    The temporal convolutional block is composed of :
    - Padding layer
    - Conv1d layer 
    - BatchNorm layer
    - ReLU layer
    - Dropout layer

    A residual connection between the input and the output of the block is present

    :param ch_in: Number of input channels
    :param ch_out: Number of output channels
    :param k_size: Kernel size
    :param dil: Amount of dilation
    :param pad: Amount of padding
    :param do_nas: Boolean variable, tells if the model should setted up for PIT-search or not
    :param nas_config: Dictionary with NAS features configurations
    """
    def __init__(self, ch_in, ch_out, k_size, dil, dropout=0.2, block_idx = None, do_nas=False, nas_config={}):
        super(TempBlock, self).__init__()
        
        self.do_nas = do_nas
        self.nas_config = nas_config

        self.exp_ratio = nas_config.get('rf_expansion_ratio', 1)
    
        self.k0 = k_size[0]
        self.k1 = k_size[1]

        self.dil0 = dil[0]
        self.dil1 = dil[1]

        self.ch_in0 = ch_in[0]
        self.ch_in1 = ch_in[1]

        self.ch_out0 = ch_out[0]
        self.ch_out1 = ch_out[1]
        
        self.block_idx = block_idx

        if not do_nas:
            if self.ch_out0 != 0:
                self.pad0 = nn.ConstantPad1d(
                    padding = ((self.k0 - 1) * dil[0], 0),
                    value = 0
                    )
                self.tcn0 = nn.Conv1d(
                        in_channels = self.ch_in0,
                        out_channels = self.ch_out0,
                        kernel_size = self.k0,
                        dilation = self.dil0,
                        bias = True
                        )
        else:
            kernel_size = ((self.k0-1)*self.dil0+1) * self.exp_ratio
            if kernel_size % 2 == 0:
                kernel_size += 1
            self.pad0 = nn.ConstantPad1d(
                padding = ((kernel_size - 1) * 1, 0),
                value = 0
                )
            self.tcn0 = PITConv1d(
                    in_channels = self.ch_in0,
                    out_channels = self.ch_out0,
                    kernel_size = kernel_size,
                    dilation = 1,
                    is_residual = True,
                    name = 'Block{}_Layer0'.format(self.block_idx),
                    prev_layer = 'Input' if self.block_idx == 0 else 'Block{}_Layer1'.format(self.block_idx - 1),
                    bias = True,
                    NAS_features = nas_config
                    )
        
        self.batchnorm0 = nn.BatchNorm1d(
                num_features = self.ch_out0
                )
        self.relu0 = nn.ReLU()
        self.dpout0 = nn.Dropout(
                p = dropout
                )
        
        if not do_nas:
            if self.ch_out0 != 0:
                self.pad1 = nn.ConstantPad1d(
                    padding = ((self.k1 - 1) * dil[1], 0),
                    value = 0
                    )
                self.tcn1 = nn.Conv1d(
                        in_channels = self.ch_in1,
                        out_channels = self.ch_out1,
                        kernel_size = self.k1,
                        dilation = self.dil1,
                        bias = True
                        )
            #if self.ch_in0 != self.ch_out1:
            if True:
                self.downsample = nn.Conv1d(
                    in_channels = self.ch_in0,
                    out_channels = self.ch_out1,
                    kernel_size = 1,
                    bias = True
                    )
                self.downsamplerelu = nn.ReLU()
                self.downsamplebn = nn.BatchNorm1d(
                        num_features = self.ch_out1
                        )
            else:
                self.downsample = None  
        else:
            kernel_size = ((self.k1-1)*self.dil1+1) * self.exp_ratio
            if kernel_size % 2 == 0:
                kernel_size += 1
            self.pad1 = nn.ConstantPad1d(
                padding = ((kernel_size - 1) * 1, 0),
                value = 0
                )
            self.tcn1 = PITConv1d(
                    in_channels = self.ch_in1,
                    out_channels = self.ch_out1,
                    kernel_size = kernel_size,
                    dilation = 1,
                    bias = True,
                    name = 'Block{}_Layer1'.format(self.block_idx),
                    prev_layer = 'Block{}_Layer0'.format(self.block_idx),
                    NAS_features = nas_config
                    )
            #if self.ch_in0 != self.ch_out1:
            if True:
                self.downsample = PITConv1d(
                    in_channels = self.ch_in0,
                    out_channels = self.ch_out1,
                    fc = True,
                    track_arch = False,
                    kernel_size = 1,
                    name = 'Block{}_LayerResConn'.format(self.block_idx),
                    prev_layer = 'Input' if self.block_idx == 0 else 'Block{}_Layer1'.format(self.block_idx - 1),
                    bias = True,
                    NAS_features = nas_config
                    )
                self.downsamplerelu = nn.ReLU()
                self.downsamplebn = nn.BatchNorm1d(
                        num_features = self.ch_out1
                        )
            else:
                self.downsample = None
            
            self.downsample.alpha = self.tcn1.alpha
            
        self.batchnorm1 = nn.BatchNorm1d(
                num_features = self.ch_out1
                )    
        self.relu1 = nn.ReLU()
        self.dpout1 = nn.Dropout(
                p = dropout
                )

        self.reluadd = nn.ReLU()
        #self.init_weights()

    def forward(self, x):
        if self.ch_out0 != 0:
            x1 = self.relu0(
                        self.batchnorm0(
                            self.tcn0(
                                self.pad0(x)
                                )
                            )
                        )

            if not self.do_nas:
                x1 = self.dpout0(x1)
            
            x2 = self.relu1(
                        self.batchnorm1(
                            self.tcn1(
                                self.pad1(x1)
                                )
                            )
                        )
            
            if not self.do_nas:
                x2 = self.dpout1(x2)

        if self.downsample is None:
            res = x
        else:
            res = self.downsample(x)
            res = self.downsamplebn(res)
            res = self.downsamplerelu(res)

        if self.ch_out0 != 0:
            return self.reluadd(x2 + res)
        else:
            return self.reluadd(res)

    def init_weights(self):
        self.tcn0.weight.data.normal_(0, 0.01)
        self.tcn1.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

class Chomp1d(BaseModel):
    """
    Module that perform a chomping operation on the input tensor.
    It is used to chomp the amount of zero-padding added on the right of the input tensor, this operation is necessary to compute causal convolutions.
    :param chomp_size: amount of padding 0s to be removed
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
                
