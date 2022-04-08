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
from torch.nn import Conv1d
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils.utils import _single
from .STE import STEFunction
from .FiGS import logistic_sigmoid_sample
import math
import numpy as np
import copy
import pdb

class PITConv1d(Conv1d):
    """
    PIT's Conv1d implementation.
    It supports various NAS features, such as:
    - Learn optimal dilation factor
    - Learn optimal filter size
    - Learn optimal number of channels
    """
    def __init__(self, in_channels, out_channels, kernel_size, fc=False, track_arch=True, is_residual=False, name='', prev_layer='', NAS_features={}, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(PITConv1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        
        self.target = NAS_features['target']
        self.mask_type = NAS_features['mask_type']

        self.device = None

        self.hysteresis = NAS_features['hysteresis']['do']
        self.eps = NAS_features['hysteresis']['eps']

        self.max_filter_len = kernel_size 
        self.tau = NAS_features.get('tau', 0.001)

        self.stride = stride
        self.fc = fc

        self.track_arch = track_arch
        self.is_residual = is_residual
        
        self.name = name
        self.prev_layer = prev_layer
        
        self.learn_dil = self.target.get('dilation', False)
        self.learn_rf = self.target.get('filters', False)
        self.learn_ch = self.target.get('channels', False)
        
        self.mask_op = self.mask_type.get('type', 'sum')
        self.mask_dil = self.mask_type.get('dilation', 'stochastic')
        self.mask_rf = self.mask_type.get('filters', 'stochastic')
        self.mask_ch = self.mask_type.get('channels', 'stochastic')

        if self.mask_ch == 'fbnet':
            self.n_mask = self.mask_type.get('n_mask', 'all')
            self.fbnet_mask = self._init_fbnet_mask(out_channels, self.n_mask)
            self.n_mask = self.n_mask if self.n_mask != 'all' else out_channels
            self.effective_ch = None
        
        # If the granularity is not specified the value 1 is used.
        self.granularity = NAS_features.get('granularity', 1)

        self.group_bin = NAS_features.get('group_binarize', False)

        if not self.fc and self.stride == 1: 
            # Dilation
            if self.mask_dil == 'binary' or self.mask_dil == 'linear':
                init_val = 1.
            elif self.mask_dil == 'stochastic':
                if self.group_bin:
                    init_val = 1.1
                else:
                    init_val = 2.5
            self.gamma = Parameter(torch.empty(
                self._max_dil(kernel_size),
                dtype = torch.float32
                ).fill_(init_val), requires_grad=True)
            self.bin_gamma = torch.empty(
                    self._max_dil(kernel_size),
                    dtype = torch.float32
                    ).fill_(init_val)
            
            # Receptive-field
            if self.mask_rf == 'binary' or self.mask_rf == 'linear':
                init_val = 1.
            elif self.mask_rf == 'stochastic':
                init_val = 2.5
            self.beta = Parameter(torch.empty(
                math.ceil(self.max_filter_len / self.granularity) - 1,
                dtype = torch.float32
                ).fill_(init_val), requires_grad=True)
            self.bin_beta = torch.empty(
                    math.ceil(self.max_filter_len / self.granularity) - 1,
                    dtype = torch.float32
                    ).fill_(init_val) 
                
        # Channels
        if self.mask_ch == 'binary' or self.mask_ch == 'linear' or self.mask_ch == 'fbnet':
            init_val = 1.
        elif self.mask_ch == 'stochastic':
            init_val = 2.5

        if self.is_residual or self.mask_ch == 'fbnet':
            self.keep_alive_ch = 0
        else:
            self.keep_alive_ch = 1

        if self.mask_ch != 'fbnet':
            self.alpha = Parameter(torch.empty(
                out_channels - self.keep_alive_ch,
                dtype = torch.float32
                ).fill_(init_val), requires_grad=True)
            self.bin_alpha = torch.empty(
                    out_channels - self.keep_alive_ch,
                    dtype = torch.float32
                    ).fill_(init_val)
        else:
            self.alpha = Parameter(torch.empty(
                self.n_mask,
                dtype = torch.float32
                ).fill_(init_val), requires_grad=True)
            self.bin_alpha = torch.empty(
                    self.n_mask,
                    dtype = torch.float32
                    ).fill_(init_val)
   
    def forward(self, x):
        if self.device is None:
            self.device = x.device
        if not self.fc and self.stride == 1:
            if self.learn_dil:
                if self.mask_dil == 'binary':
                    if self.group_bin:
                        with torch.no_grad():
                            bin_gamma = self._binarize(self.gamma, self.bin_gamma)
                    else:
                        bin_gamma = self._binarize(self.gamma, self.bin_gamma)

                if self.mask_dil == 'stochastic':
                    if self.group_bin:
                        with torch.no_grad():
                            bin_gamma = self._stochastic_mask(self.gamma, self.tau)
                            if x.is_cuda:
                                bin_gamma = bin_gamma.cuda()
                    else:
                        bin_gamma = self._stochastic_mask(self.gamma, self.tau)
                        if x.is_cuda:
                            bin_gamma = bin_gamma.cuda()

                if self.mask_dil == 'linear':
                    bin_gamma = torch.clamp(self.gamma, 0, 1)
                    if x.is_cuda:
                        bin_gamma = bin_gamma.cuda()
                self.bin_gamma = copy.deepcopy(bin_gamma.data)
            
            if self.learn_rf:
                if self.mask_rf == 'binary':
                    if self.group_bin:
                        with torch.no_grad():
                            bin_beta = self._binarize(self.beta, self.bin_beta)
                    else:
                        bin_beta = self._binarize(self.beta, self.bin_beta)
                if self.mask_rf == 'stochastic':
                    if self.group_bin:
                        with torch.no_grad():
                            bin_beta = self._stochastic_mask(self.beta, self.tau)
                            if x.is_cuda:
                                bin_beta = bin_beta.cuda()
                    else:
                        bin_beta = self._stochastic_mask(self.beta, self.tau)
                        if x.is_cuda:
                            bin_beta = bin_beta.cuda()
                if self.mask_rf == 'linear':
                    bin_beta = torch.clamp(self.beta, 0, 1)
                    if x.is_cuda:
                        bin_beta = bin_beta.cuda()
                self.bin_beta = copy.deepcopy(bin_beta.data)
            
        if self.learn_ch:
            if self.mask_ch == 'binary':
                bin_alpha = self._binarize(self.alpha, self.bin_alpha)
            if self.mask_ch == 'stochastic':
                bin_alpha = self._stochastic_mask(self.alpha, self.tau)
                if x.is_cuda:
                    bin_alpha = bin_alpha.cuda()
            if self.mask_ch == 'linear':
                bin_alpha = torch.clamp(self.alpha, 0, 1)
                if x.is_cuda:
                    bin_alpha = bin_alpha.cuda()
            if self.mask_ch == 'fbnet':
                # Compute gumbel-softmax weights
                bin_alpha = F.gumbel_softmax(self.alpha, self.tau)
                if x.is_cuda:
                    bin_alpha = bin_alpha.cuda()
            self.bin_alpha = copy.deepcopy(bin_alpha.data)

        if (self.fc or self.stride > 1) and self.learn_ch:
            if self.mask_ch == 'fbnet':
                pruned_weight = self._channel_mask_fb(self.weight, bin_alpha)
            else:
                pruned_weight = self._channel_mask(self.weight, bin_alpha)
        elif (self.fc or self.stride > 1):
            pruned_weight = self.weight
        else:
            if (self.learn_dil) and (not self.learn_rf) and (not self.learn_ch):
                if not self.group_bin:
                    pruned_weight = self._dilation_mask(self.weight, bin_gamma)
                else:
                    if self.mask_op == 'mul':
                        pruned_weight = self._dilation_mask(self.weight, self.gamma)
                    elif self.mask_op == 'sum':
                        pruned_weight = self._dilation_mask_sum(self.weight, torch.abs(self.gamma))
            elif (not self.learn_dil) and (self.learn_rf) and (not self.learn_ch):
                if not self.group_bin:
                    pruned_weight = self._filter_mask(self.weight, bin_beta)
                else:
                    if self.mask_op == 'mul':
                        pruned_weight = self._filter_mask(self.weight, self.beta)
                    elif self.mask_op == 'sum':
                        pruned_weight = self._filter_mask_sum(self.weight, torch.abs(self.beta))
            elif (not self.learn_dil) and (not self.learn_rf) and (self.learn_ch): # Ch-Only
                if self.mask_ch == 'fbnet':
                    pruned_weight = self._channel_mask_fb(self.weight, bin_alpha)
                else:
                    pruned_weight = self._channel_mask(self.weight, bin_alpha)
            elif (self.learn_dil) and (self.learn_rf) and (not self.learn_ch):
                if not self.group_bin:
                    pruned_weight = self._filter_mask(
                        self._dilation_mask(self.weight, bin_gamma), 
                        bin_beta)
                else:
                    if self.mask_op == 'mul':
                        pruned_weight = self._filter_mask(
                            self._dilation_mask(self.weight, self.gamma), 
                            self.beta)
                    elif self.mask_op == 'sum':
                        pruned_weight = self._filter_mask_sum(
                            self._dilation_mask_sum(self.weight, torch.abs(self.gamma)), 
                            torch.abs(self.beta))
            elif (not self.learn_dil) and (self.learn_rf) and (self.learn_ch):
                if not self.group_bin:
                    pruned_weight = self._filter_mask(
                        self._channel_mask(self.weight, bin_alpha), 
                        bin_beta)   
                else:
                    if self.mask_op == 'mul':
                        pruned_weight = self._filter_mask(
                            self._channel_mask(self.weight, bin_alpha), 
                            self.beta)
                    elif self.mask_op == 'sum':
                        pruned_weight = self._filter_mask(
                            self._channel_mask_sum(self.weight, bin_alpha), 
                            torch.abs(self.beta))
            elif (self.learn_dil) and (not self.learn_rf) and (self.learn_ch):
                if not self.group_bin:
                    pruned_weight = self._dilation_mask(
                        self._channel_mask(self.weight, bin_alpha), 
                        bin_gamma)   
                else:
                    if self.mask_op == 'mul':
                        pruned_weight = self._dilation_mask(
                            self._channel_mask(self.weight, bin_alpha), 
                            self.gamma)
                    elif self.mask_op == 'sum':
                        pruned_weight = self._dilation_mask_sum(
                            self._channel_mask(self.weight, bin_alpha), 
                            torch.abs(self.gamma))
            elif (self.learn_dil) and (self.learn_rf) and (self.learn_ch):
                if not self.group_bin:
                    pruned_weight = self._filter_mask(
                        self._dilation_mask(
                            self._channel_mask(self.weight, bin_alpha), 
                            bin_gamma), 
                        bin_beta)
                else:
                    if self.mask_op == 'mul':
                        pruned_weight = self._filter_mask(
                            self._dilation_mask(
                                self._channel_mask(self.weight, bin_alpha), 
                                self.gamma), 
                            self.beta)
                    elif self.mask_op == 'sum':
                        pruned_weight = self._filter_mask_sum(
                            self._dilation_mask_sum(
                                self._channel_mask(self.weight, bin_alpha), 
                                torch.abs(self.gamma)), 
                            torch.abs(self.beta))
        
        return self._conv_forward(x, pruned_weight, self.bias)
    
    def _conv_forward(self, input, weight, bias=None):
        if self.padding_mode != 'zeros':
            return F.conv1d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode), weight, bias, self.stride, _single(0), self.dilation, self.groups)
        return F.conv1d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def _binarize(self, x, bin_x):
        if self.hysteresis:
            eps_tensor = ((bin_x <= 0) * self.eps).to(self.device)
        else:
            eps_tensor = torch.zeros(x.size(), device=self.device)
        return STEFunction.apply(x, eps_tensor)

    def _stochastic_mask(self, x, tau):
        return logistic_sigmoid_sample(x, tau)
    
    def _dilation_mask(self, weight, gamma):
        eps = 1e-8
        kernel_size = weight.size()[-1]
        n_max = math.floor(math.log(kernel_size - eps, 2))
        dil_fact_max = 2 ** n_max

        matrix_list = list()
        sum_list = list()

        i = 0
        while i < kernel_size:
            vector_list = list()

            # First element and multiples of dil_fact_max are always not pruned
            if i % dil_fact_max == 0:
                vector_list.extend([0] * n_max)
                matrix_list.append(vector_list)
                sum_list.append(1)
                i += 1
            else:
                for line in self._gamma_mul_mask(dil_fact_max):
                    matrix_list.append(line)
                    sum_list.append(0)
                    i += 1

        # Truncate non-necessary rows in matrix_list, i.e., from kernel_size to end
        if len(matrix_list) != kernel_size:
            matrix_list = matrix_list[:-(len(matrix_list)-kernel_size)]
            # Same for sum_list
            sum_list = sum_list[:-(len(sum_list)-kernel_size)]

        mask_mul = torch.flip(torch.transpose(
                torch.tensor(matrix_list, dtype=torch.float32, device=self.device),
                0,
                1
                ).unsqueeze(0), (0, -1)).squeeze()
        mask_sum = torch.flip(torch.tensor(sum_list, dtype=torch.float32, device=self.device).unsqueeze(0), (0, 1)).squeeze()
        
        m_1 = torch.from_numpy(
                np.flip(
                    np.triu(
                        np.ones((n_max, n_max))),
                        1
                        ).astype('float32')
                ).to(self.device)

        m_2 = torch.from_numpy(
                np.flip(
                    np.tril(
                        np.ones((n_max, n_max)),
                        -1
                        ),
                    1
                    ).astype('float32')
                ).to(self.device)
        ones = torch.ones(1, gamma.size()[0], device=self.device)

        Gamma = torch.add(
                torch.prod(
                    torch.matmul(
                        torch.add(
                            torch.mul(
                                torch.matmul(
                                    gamma.unsqueeze(1),
                                    ones
                                    ),
                                m_1
                                ),
                            m_2
                            ),
                        mask_mul
                        ),
                    dim=0
                    ),
                mask_sum)

        if self.group_bin:
            if self.mask_dil == 'stochastic':
                return torch.mul(
                        self._stochastic_mask(Gamma, self.tau),
                        weight
                        )
            else:
                return torch.mul(
                        self._binarize(Gamma, self.bin_gamma),
                        weight
                        )
        else:
            return torch.mul(
                    Gamma,
                    weight
                    )

    def _dilation_mask_sum(self, weight, gamma):
        eps = 1e-8
        kernel_size = weight.size()[-1]
        n_max = math.floor(math.log(kernel_size - eps, 2))
        dil_fact_max = 2 ** n_max

        matrix_list = list()
        sum_list = list()

        i = 0
        while i < kernel_size:
            vector_list = list()

            # First element and multiples of dil_fact_max are always not pruned
            if i % dil_fact_max == 0:
                vector_list.extend([0] * n_max)
                matrix_list.append(vector_list)
                sum_list.append(1)
                i += 1
            else:
                for line in self._gamma_sum_mask(dil_fact_max):
                    matrix_list.append(line)
                    sum_list.append(0)
                    i += 1
        
        # Truncate non-necessary rows in matrix_list, i.e., from kernel_size to end
        if len(matrix_list) != kernel_size:
            matrix_list = matrix_list[:-(len(matrix_list)-kernel_size)]
            # Same for sum_list
            sum_list = sum_list[:-(len(sum_list)-kernel_size)]

        mask_mul = torch.flip(torch.transpose(
                torch.tensor(matrix_list, dtype=torch.float32, device=self.device),
                0,
                1
                ).unsqueeze(0), (0, -1)).squeeze()
        mask_sum = torch.flip(torch.tensor(sum_list, dtype=torch.float32, device=self.device).unsqueeze(0), (0, 1)).squeeze()

        if len(mask_mul.size()) == 1:
            mask_mul = mask_mul.unsqueeze(0)
        Gamma = torch.add(
                    torch.matmul(
                        torch.transpose(mask_mul, 0, 1),
                        gamma
                    ),
                    mask_sum
                )

        if self.group_bin:
            if self.mask_dil == 'stochastic':
                return torch.mul(
                        self._stochastic_mask(Gamma, self.tau),
                        weight
                        )
            else:
                return torch.mul(
                        self._binarize(Gamma, self.bin_gamma),
                        weight
                        )
        else:
            return torch.mul(
                    Gamma,
                    weight
                    )
    
    def _gamma_mul_mask(self, dil_fact, it=0, line=[]):

        # Recursion entry-point
        if it == 0:
            line = list()
            line.extend([[1]])
            it += 1
        # Recursion exit-point
        elif it == int(math.log(dil_fact, 2)):
            return line
        # Recursion
        else:
            for pos in range(len(line)):
                line[pos].append(0)

            line.extend([[0]*(it) + [1]])
            line.extend(copy.deepcopy(line[:(2**it-1)]))

            it += 1

        return self._gamma_mul_mask(dil_fact, it, line)

    def _gamma_sum_mask(self, dil_fact, it=0, line=[]):
        # Recursion entry-point
        if it == 0:
            line = list()
            line.extend([[1] + [0]*int(math.log(dil_fact, 2)-1)])
            it += 1
        # Recursion exit-point
        elif it == int(math.log(dil_fact, 2)):
            return line
        # Recursion
        else:
            it += 1
            line.extend([[1]*(it) + [0]*(int(math.log(dil_fact, 2))-it)])
            line.extend(line[:-1])

        return self._gamma_sum_mask(dil_fact, it, line)

    def _init_fbnet_mask(self, ch_out, n_mask):
        if n_mask == 'all':
            n_mask = ch_out
        granularity = ch_out // n_mask
        mask = []
        for i in range(n_mask):
            leading_1s = ch_out - granularity * i
            trailing_0s = ch_out - leading_1s
            ones = torch.ones(leading_1s, dtype=torch.float32, device=self.device)
            zeros = torch.zeros(trailing_0s, dtype=torch.float32, device=self.device)
            mask.append(torch.cat((ones, zeros), 0))
        return torch.stack(mask).to(self.device)

    def _fbnet_mask(self, x, tau):
        gs = F.gumbel_softmax(x, tau, dim=0)
        return gs


    def _filter_mask(self, weight, beta):
        kernel_size = weight.size()[-1]

        ones = torch.ones(1, kernel_size, device=self.device)
        m_1 = self._build_filter_matrix(kernel_size, beta.size()[0], self.granularity)
        m_2 = torch.ones(m_1.size()[0], m_1.size()[1], device=self.device) - m_1

        Beta = torch.prod(
            torch.add(
                torch.mul(
                    torch.matmul(
                        beta.unsqueeze(1),
                        ones
                    ),
                    m_1
                ),
                m_2
            ),
            dim = 0
        )

        if self.group_bin:
            if self.mask_dil == 'stochastic':
                return torch.mul(
                        self._stochastic_mask(Beta, self.tau),
                        weight
                        )
            else:
                return torch.mul(
                    self._binarize(Beta, self.bin_beta),
                    weight
                )
        else:
            return torch.mul(
                Beta,
                weight
            )

    def _filter_mask_sum(self, weight, beta):
        kernel_size = weight.size()[-1]

        m_1 = self._build_filter_matrix(kernel_size, beta.size()[0], self.granularity)
        m_1 = torch.cat((torch.flip(torch.transpose(m_1[:, :-1], 0, 1).unsqueeze(0), [0, 1]).squeeze(0), m_1[:, -1].unsqueeze(0)))
        m_2 = torch.zeros(kernel_size, device=self.device)
        m_2[-1] = 1.
        
        Beta = torch.add(
                    torch.matmul(
                        m_1,
                        beta
                        ),
                    m_2
                    )

        if self.group_bin:
            if self.mask_dil == 'stochastic':
                return torch.mul(
                        self._stochastic_mask(Beta, self.tau),
                        weight
                        )
            else:
                return torch.mul(
                    self._binarize(Beta, self.bin_beta),
                    weight
                )
        else:
            return torch.mul(
                Beta,
                weight
            )

    def _build_filter_matrix(self, length, rows, granularity):
        matrix = list()
        line = list()

        for i in range(rows):
            leading_1 = length - 1 - i*granularity
            trailing_0 = length - leading_1  
            line.extend(
                [1] * leading_1 +
                [0] * trailing_0 
                )
            matrix.append(line)
            line = list()

        return torch.tensor(matrix, dtype=torch.float32, device=self.device)

    def _channel_mask(self, weight, alpha):
        # Weight format in Conv1d is [out_channels, in_channels, kernel]
        masked_channels = torch.mul(
            # First channel is pruned or not depending on the value of keep_alive_ch
            weight.transpose(0, 2)[:, :, self.keep_alive_ch:],
            alpha
            )
        if not self.is_residual:
            return torch.cat((weight.transpose(0, 2)[:, :, 0].unsqueeze(-1), masked_channels), -1).transpose(0, 2)
        else:
            return masked_channels.transpose(0, 2)

    def _channel_mask_fb(self, weight, alpha):
        ch_out = weight.size()[0]
        # Multiply gumbel-softmax weight to fbnet masks and sum them
        curr_mask = sum([self.fbnet_mask[i].to(self.device) * alpha[i] for i in range(self.n_mask)])
        self.effective_ch = torch.sum(curr_mask) # Used to compute reg loss
        # Multiply current mask to weight tensor
        return weight * curr_mask.view(ch_out, 1, 1)
        
    def _max_dil(self, kernel_size):
        """
        Compute the largest allowed dilation factor given the kernel_size
        """
        eps = 1e-8
        return math.floor(math.log(kernel_size - eps,2))

