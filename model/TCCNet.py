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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from base import BaseModel
from torch.nn.parameter import Parameter
from NAS.layer.PITConv1d_v2 import PITConv1d
from math import ceil
import pdb

class TCCNet(BaseModel):
    """
    TCCNet architecture:
    Variable number of repeated instances of TemporalBlock, plus a final residual block with a single convolution.
    At the end of the network is possible to find an Attention layer or a MeanOverTime layer.
    """
    def __init__(self, dataset_name='NinaProDB1', dataset_args={}, do_nas=False, nas_config={}, learned_dil=[], learned_rf=[], learned_ch=[]):
        super(TCCNet, self).__init__()
        
        self.do_nas = do_nas
        self.nas_config = nas_config
        self.dataset_name = dataset_name
        self.dataset_args = dataset_args
        self.num_channels = dataset_args['num_channels']
        self.exp_ratio = nas_config.get('rf_expansion_ratio', 1)
        
        self.tcn = TempNet(
                num_inputs = self.dataset_args['input_size'], 
                num_channels = self.num_channels, 
                kernel_size = self.dataset_args['kernel_size'], 
                dropout = self.dataset_args['dropout'], 
                learned_dil = learned_dil, 
                learned_rf = learned_rf, 
                learned_ch = learned_ch,
                do_nas = do_nas, 
                nas_config = nas_config
                )

        dil_last = 64 if not learned_dil else learned_dil[-1]
        rf_last = 129 if not learned_rf else learned_rf[-1]
        ch_last = 128 if not learned_ch else learned_ch[-1]

        dil_last = 1 if (learned_rf) and (not learned_dil) else dil_last

        if not do_nas:
            self.pad_last = nn.ConstantPad1d(
                padding = ((ceil(rf_last / dil_last) - 1) * dil_last, 0),
                value = 0
                )
            self.tcn_last = nn.Conv1d(
                    in_channels = self.num_channels[-1] if not learned_ch else learned_ch[-2],
                    out_channels = ch_last,
                    kernel_size = ceil(rf_last / dil_last),
                    dilation = dil_last#,
                    #padding = self.pad0
                    )
            self.downsample_last = nn.Conv1d(
                in_channels = self.num_channels[-1] if not learned_ch else learned_ch[-2],
                out_channels = ch_last,
                kernel_size = 1
                )
        else:
            kernel_size = ((self.dataset_args['kernel_size']-1)*dil_last+1)
            self.pad_last = nn.ConstantPad1d(
                padding = ((kernel_size - 1) * 1, 0),
                value = 0
                )
            self.tcn_last = PITConv1d(
                    in_channels = self.num_channels[-1],
                    out_channels = 128,
                    kernel_size = kernel_size,
                    dilation = 1,
                    name = 'BlockLast_Layer0',
                    prev_layer = 'Block{}_Layer1'.format(len(self.num_channels) - 1),
                    #padding = (self.k0-1)*self.dil0,
                    NAS_features = nas_config
                    )
            self.downsample_last = PITConv1d(
                in_channels = self.num_channels[-1],
                out_channels = 128,
                kernel_size = 1,
                fc = True,
                name = 'BlockLast_LayerResConn',
                prev_layer = 'Block{}_Layer1'.format(len(self.num_channels) - 1),
                track_arch = False,
                NAS_features = nas_config
                )
            # Force alphas of tcn_last and downsample_last to be equal
            self.downsample_last.alpha = self.tcn_last.alpha
            #self.downsample_last = nn.Conv1d(
            #    in_channels = self.num_channels[-1],
            #    out_channels = 128,
            #    kernel_size = 1
            #    )
        self.relu_last = nn.ReLU()
        self.dpout_last = nn.Dropout2d(
                p = self.dataset_args['dropout'] 
                )

        if self.dataset_args['classify_as'] == 'aot':
            self.classifier = MeanOverTime()
        elif self.dataset_args['classify_as'] == 'att':
            self.classifier = AttentionWithContext(ch_last)

        self.out = nn.Linear(
                in_features = ch_last,
                out_features = self.dataset_args['output_size']
                )
        
    def forward(self, x):
        x = x.transpose(1,2)
        y1 = self.tcn(x)
        
        y2 = self.relu_last(
                self.tcn_last(
                    self.pad_last(y1)
                    )
                )

        if not self.do_nas:   
            y2_dpout = self.dpout_last(y2.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            # turn-off dpout during nas search
            y2_dpout = y2
            #y2_dpout = self.dpout_last(y2.permute(0, 2, 1)).permute(0, 2, 1)

        skip = self.downsample_last(y1)
        
        y3 = y2_dpout + skip

        out = self.out(
            self.classifier(
                y3
            )
        )

        return out

class TempNet(BaseModel):
    """
    Temporal Convolutional Net composed of a number of TempBlock defined with a specific parameter.
    """
    def __init__(self, num_inputs, num_channels, kernel_size, dropout=0.05, learned_dil=[], learned_rf=[], learned_ch=[], do_nas=False, nas_config={}):
        super(TempNet, self).__init__()
        layers = list()
        num_levels = len(num_channels)
        original_rf = [(kernel_size - 1) * (2**i) + 1 for i in range(2*num_levels)]
        for i in range(num_levels):
            dilation_size = list()
            k = list()
            ch_in = list()
            ch_out = list()
            if not learned_dil:
                dilation_size.append(2 ** (2*i))
                dilation_size.append(2 ** (2*i+1))
            else:
                dilation_size.append(learned_dil[2*i])
                dilation_size.append(learned_dil[2*i+1])
            if not learned_rf:
                k.append(ceil(original_rf[2*i]/dilation_size[0]))
                k.append(ceil(original_rf[2*i+1]/dilation_size[1]))
            elif not learned_dil:
                dilation_size[0] = 1
                dilation_size[1] = 1
                k.append(ceil(learned_rf[2*i]/dilation_size[0]))
                k.append(ceil(learned_rf[2*i+1]/dilation_size[1]))
            else:
                k.append(ceil(learned_rf[2*i]/dilation_size[0]))
                k.append(ceil(learned_rf[2*i+1]/dilation_size[1]))
            if not learned_ch:
                ch_in.append(num_inputs if i == 0 else num_channels[i-1])
                ch_in.append(num_channels[i])
                ch_out.append(num_channels[i])
                ch_out.append(num_channels[i])
            else:
                ch_in.append(num_inputs if i == 0 else learned_ch[2*i-1])
                ch_in.append(learned_ch[0] if i == 0 else learned_ch[2*i])
                ch_out.append(learned_ch[2*i])
                ch_out.append(learned_ch[2*i+1])
            
            layers += [TempBlock(
                ch_in = ch_in, 
                ch_out = ch_out, 
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
    - ReLU layer
    - Spatial Dropout layer

    A residual connection between the input and the output of the block is present

    :param ch_in: Number of input channels
    :param ch_out: Number of output channels
    :param k_size: Kernel size
    :param dil: Amount of dilation
    :param pad: Amount of padding
    :param do_nas: Boolean variable, tells if the model should setted up for PIT-search or not
    :param nas_config: Dictionary with NAS features configurations
    """
    def __init__(self, ch_in, ch_out, k_size, dil, dropout=0.05, block_idx=None, do_nas=False, nas_config={}):
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
                    padding = ((self.k0 - 1) * self.dil0, 0),
                    value = 0
                    )
                self.tcn0 = nn.Conv1d(
                        in_channels = self.ch_in0,
                        out_channels = self.ch_out0,
                        kernel_size = self.k0,
                        dilation = self.dil0#,
                        #padding = self.pad0
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
                    #padding = (self.k0-1)*self.dil0,
                    NAS_features = nas_config
                    )
        self.relu0 = nn.ReLU()
        self.dpout0 = nn.Dropout2d(
                p = dropout
                )

        if not do_nas:
            if self.ch_out0 != 0:
                self.pad1 = nn.ConstantPad1d(
                    padding = ((self.k1 - 1) * self.dil1, 0),
                    value = 0
                    )
                self.tcn1 = nn.Conv1d(
                        in_channels = self.ch_in1,
                        out_channels = self.ch_out1,
                        kernel_size = self.k1,
                        dilation = self.dil1#,
                        #padding = self.pad1
                        )
            self.downsample = nn.Conv1d(
                in_channels = self.ch_in0,
                out_channels = self.ch_out1,
                kernel_size = 1
                )
                    
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
                name = 'Block{}_Layer1'.format(self.block_idx),
                prev_layer = 'Block{}_Layer0'.format(self.block_idx),
                #padding = (self.k1-1)*self.dil1,
                NAS_features = nas_config
                )
            self.downsample = PITConv1d(
                in_channels = self.ch_in0,
                out_channels = self.ch_out1,
                fc = True,
                track_arch = False,
                kernel_size = 1,
                name = 'Block{}_LayerResConn'.format(self.block_idx),
                prev_layer = 'Input' if self.block_idx == 0 else 'Block{}_Layer1'.format(self.block_idx - 1),
                NAS_features = nas_config
                )
            self.downsample.alpha = self.tcn1.alpha
            
        self.relu1 = nn.ReLU()
        self.dpout1 = nn.Dropout2d(
                p = dropout
                )

    def forward(self, x):
        if self.ch_out0 != 0:
            x1 = self.relu0(
                    self.tcn0(
                        self.pad0(x)
                        )
                    )
            if not self.do_nas:
                x1_drop = self.dpout0(x1.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                x1_drop = x1
                #x1_drop = self.dpout0(x1.permute(0, 2, 1)).permute(0, 2, 1)

            x2 = self.relu1(
                    self.tcn1(
                        self.pad1(x1_drop)
                        )
                    )
            
            if not self.do_nas:
                x2_drop = self.dpout1(x2.permute(0, 2, 1)).permute(0, 2, 1) 
            else:
                x2_drop = x2
                #x2_drop = self.dpout1(x2.permute(0, 2, 1)).permute(0, 2, 1) 

        res = self.downsample(x)

        if self.ch_out0 != 0:
            return x2_drop + res
        else:
            return res

    def init_weights(self):
        self.tcn0.weight.data.normal_(0, 0.01)
        self.tcn1.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

class MeanOverTime(BaseModel):
    """
    Layer that computes the mean of timesteps.
    """
    def __init__(self):
        super(MeanOverTime, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=2)

class AttentionWithContext(nn.Module):
    """
    Attention operation, with a context/query vector, for temporal data.
    """
    def __init__(self, in_channels):
        super(AttentionWithContext, self).__init__()
        
        self.in_channels = in_channels

        self.Wa = Parameter(torch.Tensor(in_channels, in_channels))
        nn.init.xavier_uniform_(self.Wa)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        self.u = Parameter(torch.Tensor(in_channels))
        nn.init.xavier_uniform_(self.u.unsqueeze(1))

    def forward(self, x):
        v = torch.matmul(self.Wa, x)
        v = self.tanh(v)
        prod = torch.matmul(self.u, v)
        prod = self.softmax(prod)
        
        return torch.sum(x*prod.unsqueeze(1), dim=2)

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
                
