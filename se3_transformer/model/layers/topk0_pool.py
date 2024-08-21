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
from dgl import DGLGraph
from dgl.nn.pytorch import AvgPooling, MaxPooling
from torch import Tensor
from dgl import backend as F
import torch
from se3_transformer.model.fiber import Fiber
import numpy as np


def topK_se3(graph, feat, xi, k):
    #remove this read from graph code, since se3 transformer natively uses pulled out feats from graph
    # READOUT_ON_ATTRS = {
    #     "nodes": ("ndata", "batch_num_nodes", "number_of_nodes"),
    #     "edges": ("edata", "batch_num_edges", "number_of_edges"),
    # }
    # _, batch_num_objs_attr, _ = READOUT_ON_ATTRS["nodes"]

    # #this is a fancy way of saying 'batch_num_nodes
    # data = getattr(bg, "nodes")[None].data
    # if F.ndim(data[feat]) > 2:
    #     raise DGLError(
    #         "Only support {} feature `{}` with dimension less than or"
    #         " equal to 2".format(typestr, feat)
    #     )
    # feat = data[feat]


    hidden_size = feat.shape[-1]
    batch_num_objs = getattr(graph, 'batch_num_nodes')(None)
    batch_size = len(batch_num_objs)
    descending = True

    length = max(max(F.asnumpy(batch_num_objs)), k) #max k or batch of nodes size
    fill_val = -float("inf") if descending else float("inf")
    
    feat_y = F.pad_packed_tensor(
        feat, batch_num_objs, fill_val, l_min=k
    )  # (batch_size, l, d)

    order = F.argsort(feat_y, 1, descending=descending)
    topk_indices_unsort_batch = F.slice_axis(order, 1, 0, k)
    #sort to matches original connectivity with define_graph_edges, likely change but probably won't hurt now
    topk_indices, tpk_ind = torch.sort(topk_indices_unsort_batch,dim=1) 

    #get batch shifts
    feat_ = F.reshape(feat_y, (-1,))
    shift = F.repeat(
        F.arange(0, batch_size), k * hidden_size, -1
    ) * length * hidden_size + F.cat(
        [F.arange(0, hidden_size)] * batch_size * k, -1
    )
    
    shift = F.copy_to(shift, F.context(feat))
    topk_indices_ = F.reshape(topk_indices, (-1,)) * hidden_size + shift
    #trainable params gather
    out_y = F.reshape(F.gather_row(feat_, topk_indices_), (batch_size*k, -1))
    out_y = F.replace_inf_with_zero(out_y)
    #nodes features gather
    out_xi = F.reshape(F.gather_row(xi, topk_indices_), (batch_size*k, -1))
    out_xi = F.replace_inf_with_zero(out_xi)
    return out_y, out_xi, topk_indices_


class TopK_Pool(torch.nn.Module):
    """
    https://arxiv.org/pdf/1905.05178.pdf
    Project Node Features to 1D for topK pooling using trainable weights
    #code from Linear Layer SE3 add topK_se3 method
    Only type '0' features coded so far, no interactions between types on linear layers
    """
    
    #in the future can I pool '1' features

    def __init__(self, fiber_in: Fiber, k=5):
        super().__init__()
        self.k = k
        fiber_out = Fiber({0: 1}) #convert to 1D of nodes for topK selection
        self.weights = torch.nn.ParameterDict({
            str(degree_out): torch.nn.Parameter(
                torch.randn(channels_out, fiber_in[degree_out]) / np.sqrt(fiber_in[degree_out]))
            for degree_out, channels_out in fiber_out
        })
        


    def forward(self, features: Dict[str, Tensor], graph: DGLGraph) -> Dict[str, Tensor]:
        #add topK selection, sigmoid, return nodes
        yi = {
            degree: torch.div(self.weights[degree] @ features[degree], self.weights[degree].norm())
            for degree, weight in self.weights.items()
        }
        y_selected, feats_selected, topk_indices_batched = topK_se3(graph, yi['0'], features['0'], self.k)
        return {'0':(torch.sigmoid(y_selected)*feats_selected).unsqueeze(-1)}, topk_indices_batched