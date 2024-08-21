# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

from typing import Dict, Literal

import torch.nn as nn
import torch
from se3_transformer.model.fiber import Fiber
from dgl import DGLGraph
from dgl.nn.pytorch import AvgPooling, MaxPooling
from dgl import backend as F
from torch import Tensor


class GPooling(nn.Module):
    """
    Graph max/average pooling on a given feature type.
    The average can be taken for any feature type, and equivariance will be maintained.
    The maximum can only be taken for invariant features (type 0).
    If you want max-pooling for type > 0 features, look into Vector Neurons.
    """

    def __init__(self, feat_type: int = 0, pool: Literal['max', 'avg'] = 'max'):
        """
        :param feat_type: Feature type to pool
        :param pool: Type of pooling: max or avg
        """
        super().__init__()
        assert pool in ['max', 'avg'], f'Unknown pooling: {pool}'
        assert feat_type == 0 or pool == 'avg', 'Max pooling on type > 0 features will break equivariance'
        self.feat_type = feat_type
        self.pool = MaxPooling() if pool == 'max' else AvgPooling()

    def forward(self, features: Dict[str, Tensor], graph: DGLGraph, **kwargs) -> Tensor:
        pooled = self.pool(graph, features[str(self.feat_type)])
        return pooled.squeeze(dim=-1)
    
class Latent_Unpool(torch.nn.Module):
    """
    Duplicate Latent onto Graph with k_nodes. Add across equivalent down features from U-net
    :param fiber_in: Fiber size of latent
    :param fiber_add: Fiber size of equivalent down features to add
    :param k_nodes: Number of times to replicate latent onto graph of size k (from top_k pooling)
    """

    def __init__(self, fiber_in: Fiber, fiber_add: Fiber, knodes: int):
        super().__init__()
        self.fiber_out = Fiber.combine_max(fiber_in, fiber_add)
        self.node_repeat = knodes

    def forward(self, features: Dict[str, Tensor], u_features: Dict[str, Tensor]):
        out_feats = {}
        for degree_out, channels_out in self.fiber_out:
            cd = str(degree_out)
            if cd in features.keys():
                #repeat latent for all nodes
                feat_out = features[cd].repeat_interleave(self.node_repeat,0)[...,None]
                #add upper level features to front of the tensor the repeated latent tensor (if add is smaller size)
                out_feats[cd] = torch.add(feat_out, 
                                          torch.concat((u_features[cd],
                                                       torch.zeros((u_features[cd].shape[0],)+
                                                                   (features[cd].shape[1]-u_features[cd].shape[1],)+
                                                                   u_features[cd].shape[2:]
                                                                   ,device = u_features[cd].device)
                     ),axis=1))

            else:
                #upper feats have additional type features, copy over
                out_feats[cd] = u_features[cd]
                    
        return out_feats
    
class Unpool_Layer(torch.nn.Module):
    """
    Uses indices to place nodes into zeros. Add equivalent features from down side of U-net
    Assumes lower features are more
    :param fiber_in: Fiber size of features to unpool
    :param fiber_add: Fiber size of equivalent down features to add to unpooled features 
                       (set onto zero vector by saved indices from pooling)
    """

    def __init__(self, fiber_in: Fiber, fiber_add: Fiber):
        super().__init__()
        self.fiber_in = fiber_in
        self.fiber_add = fiber_add
        self.fiber_out = Fiber.combine_max(fiber_in, fiber_add)
        
    def forward(self, features: Dict[str, Tensor], u_features: Dict[str, Tensor], idx : Tensor):
        out_feats = {}
        for degree_out, channels_out in self.fiber_out:
            cd = str(degree_out)
            unpool_feats = u_features[cd].new_zeros([u_features[cd].shape[0], features[cd].shape[1], u_features[cd].shape[2]])
            pad = features[cd].new_zeros([unpool_feats.shape[0], unpool_feats.shape[1]-u_features[cd].shape[1], unpool_feats.shape[2]])
            out_feats[cd] = torch.add(F.scatter_row(unpool_feats,idx,features[cd]), torch.cat((u_features[cd],pad),1))

        return out_feats
    

        
