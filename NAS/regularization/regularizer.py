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
import numpy as np
from math import ceil, log, floor
from NAS.layer.STE import STEFunction
import pdb

class Regularizer:
    def __init__(self, prev_layer, layer, mask_type, metric, nas_target, reg_target, output_shape, device, nas_config={}):
        self.prev_layer = prev_layer
        self.layer = layer
        self.mask_type = mask_type
        self.metric = metric
        self.nas_target = nas_target
        # If reg_target is different than None it means that we are performing the search targeting a specific hyper-parameter
        # and the others are freezed, thus their binarized value need to be used in the regularization expression
        self.reg_target = reg_target
        self.device = device
        self.nas_config = nas_config
        self.output_shape = output_shape[-1]

        self._ch_in = getattr(prev_layer, 'weight').size()[1]
        self._ch_out = getattr(layer, 'weight').size()[0]
        self._rf = getattr(layer, 'weight').size()[-1]
        
        self.learn_dil = nas_target['dilation']

        self.learn_rf = nas_target['filters']
        self.gr = nas_config['granularity'] #if self.learn_rf else 1

        self.learn_ch = nas_target['channels']

        self.mask_op = self.mask_type.get('type', 'sum')
        self.mask_dil = mask_type.get('dilation', 'binary')
        self.mask_rf = mask_type.get('filters', 'binary')
        self.mask_ch = mask_type.get('channels', 'binary')

    @property
    def reg_loss(self):
        if self.metric == 'size' or self.metric == 'flops':
            return self._eval_reg_loss()
        else:
            raise ValueError("Possible metrics that can be regularized are: 'size' or 'flops'")
    
    def _eval_reg_loss(self):
        if (self.reg_target is not None) and (self.reg_target != 'alpha'):
            ch_in = torch.sum(
                torch.abs(getattr(self.prev_layer, 'bin_alpha'))
            )
            ch_out = torch.sum(
                torch.abs(getattr(self.layer, 'bin_alpha'))
            )
        else:
            ch_in = torch.sum(
                    torch.abs(getattr(self.prev_layer, 'alpha'))
                )
            ch_out = torch.sum(
                torch.abs(getattr(self.layer, 'alpha'))
            )
        
        if self.layer.prev_layer == 'Input':
            ch_in = self._ch_in

        if self.layer.fc: # or pointwise conv
            return ch_in * ch_out
        elif self.layer.stride > 1:
            return ch_in * ch_out * self._rf
        else:
            if ((self.reg_target is not None) and (self.reg_target != 'beta')) or (not self.learn_rf):
                if self.mask_op == 'mul':
                    Beta = torch.prod(
                    self._aggregate_nas_parameter(torch.abs(getattr(self.layer, 'bin_beta'))),
                    0
                    )
                elif self.mask_op == 'sum':
                    Beta = self._aggregate_nas_parameter_sum(torch.abs(getattr(self.layer, 'bin_beta')))
                    norm_beta = torch.tensor([float(i) for i in reversed(range(1, Beta.size()[0]+1))], device=self.device)
                    Beta = Beta / norm_beta
            else:
                if self.mask_op == 'mul':
                    Beta = torch.prod(
                        self._aggregate_nas_parameter(torch.abs(getattr(self.layer, 'beta'))),
                        0
                        )
                elif self.mask_op == 'sum':
                    Beta = self._aggregate_nas_parameter_sum(torch.abs(getattr(self.layer, 'beta')))
                    norm_beta = torch.tensor([float(i) for i in reversed(range(1, Beta.size()[0]+1))], device=self.device)
                    Beta = Beta / norm_beta

            Beta_0 = torch.ones(1, requires_grad=False, device=self.device) 
            Beta = torch.cat((Beta_0, Beta))
            
            if ((self.reg_target is not None) and (self.reg_target != 'gamma')) or (not self.learn_dil):
                if self.mask_op == 'mul':
                    Gamma = torch.prod(
                        self._aggregate_nas_parameter(torch.abs(getattr(self.layer, 'bin_gamma'))),
                        0
                        )
                elif self.mask_op == 'sum':
                    Gamma = self._aggregate_nas_parameter_sum(torch.abs(getattr(self.layer, 'bin_gamma')))
                    norm_gamma = torch.tensor([float(i) for i in reversed(range(1, Gamma.size()[0]+1))], device=self.device)
                    Gamma = Gamma / norm_gamma
            else:
                if self.mask_op == 'mul':
                    Gamma = torch.prod(
                        self._aggregate_nas_parameter(torch.abs(getattr(self.layer, 'gamma'))),
                        0
                        )
                elif self.mask_op == 'sum':
                    Gamma = self._aggregate_nas_parameter_sum(torch.abs(getattr(self.layer, 'gamma')))
                    norm_gamma = torch.tensor([float(i) for i in reversed(range(1, Gamma.size()[0]+1))], device=self.device)
                    Gamma = Gamma / norm_gamma

            Gamma_0 = torch.ones(1, requires_grad=False, device=self.device)
            Gamma = torch.cat((Gamma_0, Gamma))

            Beta_index = self._compute_beta_index()
            Gamma_index = self._compute_gamma_index()
            Beta_Gamma = torch.sum(torch.mul(
                torch.gather(Beta, 0, Beta_index),
                torch.gather(Gamma, 0, Gamma_index)
            ))

            if self.metric == 'size':
                return ch_in * ch_out * Beta_Gamma
            elif self.metric == 'flops':
                return ch_in * ch_out * Beta_Gamma * self.output_shape

    def _compute_beta_index(self):
        index = [ceil(i / self.gr) for i in range(self._rf)]
        return torch.tensor(index, device=self.device)
    
    def _compute_gamma_index(self):
        index = [sum([i % 2**p != 0 for p in range(1, floor(log(self._rf, 2))+1)]) for i in range(self._rf)]
        return torch.tensor(index, device=self.device)

    def _aggregate_nas_parameter(self, nas_parameter):
        dim = nas_parameter.size()[0]
        ones = torch.ones(1, dim, device=self.device)
        mask_mul = torch.from_numpy(np.transpose(np.tril(np.ones((dim, dim)), 0)).astype('float32')).to(self.device)
        mask_sum = torch.from_numpy(np.tril(np.ones((dim, dim)), -1).astype('float32')).to(self.device)

        return torch.mul(torch.matmul(nas_parameter.unsqueeze(1), ones), mask_mul) + mask_sum
    
    def _aggregate_nas_parameter_sum(self, nas_parameter):
        dim = nas_parameter.size()[0]
        mask_mul = torch.from_numpy(np.flip(np.triu(np.ones((dim, dim))), -1).astype('float32')).to(self.device)

        return torch.matmul(mask_mul, nas_parameter.to(self.device))