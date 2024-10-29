import torch
import numpy as np
import util.npose_util as nu
import os
import pathlib
import dgl
from dgl import backend as F
from torch.utils.data import random_split, DataLoader, Dataset
from typing import Dict
from torch import Tensor
from dgl import DGLGraph
from torch import nn
from torch import einsum
import time

from se3_transformer.model.basis import get_basis, update_basis_with_fused
from se3_transformer.model.transformer import Sequential, SE3Transformer
from se3_transformer.model.transformer_topk import SE3Transformer_topK
from se3_transformer.model.FAPE_Loss import FAPE_loss, Qs2Rs, normQ
from se3_transformer.model.layers.attentiontopK import AttentionBlockSE3
from se3_transformer.model.layers.linear import LinearSE3
from se3_transformer.model.layers.convolution import ConvSE3, ConvSE3FuseLevel
from se3_transformer.model.layers.norm import NormSE3
from se3_transformer.model.layers.pooling import GPooling, Latent_Unpool, Unpool_Layer
from se3_transformer.runtime.utils import str2bool, to_cuda
from se3_transformer.model.fiber import Fiber
from se3_transformer.model.transformer import get_populated_edge_features

from gudiff_model.PDBDataSet_GraphCon import define_graph_edges


#helper functions for Diffusion_Graphical_UNet_Model

def define_poolGraph(n_nodes, batch_size, cast_type=torch.float32, cuda_out=True ):
    
    v1,v2,edge_data, ind = define_graph_edges(n_nodes)
    #pe = make_pe_encoding(n_nodes=n_nodes)#pe e
    
    graphList = []
    
    for i in range(batch_size):
        
        g = dgl.graph((v1,v2))
        g.edata['con'] = edge_data.type(cast_type).reshape((-1,1))
        g.ndata['pos'] = torch.zeros((n_nodes,3),dtype=torch.float32)

        graphList.append(g)
        
    batched_graph = dgl.batch(graphList)
    
    if cuda_out:
        return to_cuda(batched_graph)
    else:
        return batched_graph            
        
def pull_edge_features(graph, edge_feat_dim=1):
    return {'0': graph.edata['con'][:, :edge_feat_dim, None]}

def prep_for_gcn(graph, xyz_pos, edge_feats_input, idx, max_degree=3, comp_grad=True):
    
    src, dst = graph.edges()
    
    new_pos = F.gather_row(xyz_pos, idx)
    rel_pos = F.gather_row(new_pos,dst) - F.gather_row(new_pos,src) 
    
    basis_out = get_basis(rel_pos, max_degree=max_degree,
                                   compute_gradients=comp_grad,
                                   use_pad_trick=False)
    basis_out = update_basis_with_fused(basis_out, max_degree, use_pad_trick=False,
                                            fully_fused=False)
    edge_feats_out = get_populated_edge_features(rel_pos, edge_feats_input)
    return edge_feats_out, basis_out, new_pos    

#--- layer for converting t[0,1] to be more expressive
class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""  
    #From Yang Song, Tutorial on Score based diffusion models
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed 
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    
class GaussianFourierProjection_Linear(nn.Module):
    """Gaussian random features for encoding time steps."""  
    #From Yang Song, Tutorial on Score based diffusion models
    def __init__(self, embed_dim, scale=30., use_deg1 = True):
        super().__init__()
        self.GFP = GaussianFourierProjection(embed_dim, scale=scale)
        self.linear0 = nn.Linear(embed_dim,embed_dim)
        self.use_deg1 = use_deg1
        self.act = nn.SiLU()
        if use_deg1:
            self.linear1 = nn.Linear(embed_dim,1) #create a scalar for multiplication to vector '1'
            #according to to some toronto math notes I believe this retains invariance

    def forward(self, t):
        if self.use_deg1:
            return {'0': self.act(self.linear0(self.GFP(t))), '1':self.act(self.linear1(self.GFP(t)))}
        else:
            return {'0' : self.act(self.linear0(self.GFP(t)))}
        
##-- Graphical U-net for Denoising
class GraphUNet(torch.nn.Module):
    def __init__(self, 
                 fiber_start = Fiber({0:12, 1:2}),
                 fiber_out = Fiber({1:2}),
                 k=4,
                 batch_size = 8,
                 stride=4,
                 max_degree=3,
                 channels=32,
                 feat_0 = 32,
                 feat_1 = 6,
                 num_heads = 8,
                 channels_div=4,
                 num_layers = 1,
                 num_layers_ca = 1,
                 edge_feature_dim=1,
                 latent_pool_type = 'avg',
                 t_size = 12,
                 zero_lin=True,
                 use_tdeg1 = True,
                 cuda=True):
        super(GraphUNet, self).__init__()
        
        
        #self.comp_basis_grad = True
        self.cuda = cuda
        
        if cuda:
            self.device='cuda:0'
        else:
            self.device='cpu'
        
        self.max_degree=max_degree
        self.B = batch_size
        self.k = k
        self.ts = t_size
        self.use_tdeg1 = use_tdeg1
        self.zero_lin = zero_lin
        
        self.embed_t = GaussianFourierProjection_Linear(self.ts, use_deg1=self.use_tdeg1)
        self.t_fiber = Fiber({0:self.ts}) #add for change in fiber with self.concat_t
        
        self.num_layers = num_layers
        self.num_layers_ca = num_layers_ca
        self.channels = channels
        self.feat0 = feat_0
        self.feat1 = feat_1
        self.channels_div = channels_div
        self.num_heads = num_heads
        self.mult = int(stride/2)
        self.fiber_edge=Fiber({0:edge_feature_dim})
        self.edge_feat_dim = edge_feature_dim
        
        self.pool_type = latent_pool_type
        
        self.channels_down_ca = channels
        #down c_alpha interactions by radius
        self.fiber_start =  fiber_start
        self.fiber_hidden_down_ca = Fiber.create(self.max_degree, self.channels_down_ca)
        self.fiber_out_down_ca =Fiber({0: self.feat0, 1: self.feat1})
        
        #concat_t, plus one on input fiber, run concat_t method on forward
        self.down_ca = SE3Transformer(num_layers = self.num_layers_ca,
                        fiber_in=self.fiber_start+self.t_fiber,
                        fiber_hidden= self.fiber_hidden_down_ca, 
                        fiber_out=self.fiber_out_down_ca,
                        num_heads = self.num_heads,
                        channels_div = self.channels_div,
                        fiber_edge=self.fiber_edge,
                        low_memory=True,
                        tensor_cores=False)
        
        self.channels_down_ca2mp = self.channels_down_ca*self.mult
        
        #pool from c_alpha onto midpoints
        self.fiber_in_down_ca2mp     = self.fiber_out_down_ca
        self.fiber_hidden_down_ca2mp = Fiber.create(max_degree, self.channels_down_ca2mp)
        self.fiber_out_down_ca2mp    = Fiber({0: self.feat0*self.mult, 1: self.feat1*self.mult})
        
        #concat_t, plus one on input fiber, run concat_t method on forward
        self.down_ca2mp = SE3Transformer(num_layers = self.num_layers,
                            fiber_in     = self.fiber_in_down_ca2mp+self.t_fiber,
                            fiber_hidden = self.fiber_hidden_down_ca2mp, 
                            fiber_out    = self.fiber_out_down_ca2mp,
                            num_heads =    self.num_heads,
                            channels_div = self.channels_div,
                            fiber_edge=self. fiber_edge,
                            low_memory=True,
                            tensor_cores=False)
        
        self.fiber_in_mptopk =  self.fiber_out_down_ca2mp
        self.fiber_hidden_down_mp  =self.fiber_hidden_down_ca2mp
        self.fiber_out_down_mp_out =self.fiber_out_down_ca2mp
        self.fiber_out_topkpool=Fiber({0: self.feat0*self.mult*self.mult})
        
        #concat_t, plus one on input fiber, run concat_t method on forward
        self.mp_topk = SE3Transformer_topK(num_layers      = self.num_layers,
                                        fiber_in      = self.fiber_in_mptopk+self.t_fiber,
                                        fiber_hidden  = self.fiber_hidden_down_mp, 
                                        fiber_out     = self.fiber_out_down_mp_out ,
                                        fiber_out_topk= self.fiber_out_topkpool,
                                        k             = self.k,
                                        num_heads     = self.num_heads,
                                        channels_div  = self.channels_div,
                                        fiber_edge    =  self.fiber_edge,
                                        low_memory=True,
                                        tensor_cores=False)
        
        self.gsmall = define_poolGraph(self.k, self.B, cast_type=torch.float32, cuda_out=self.cuda)
        self.ef_small = pull_edge_features(self.gsmall, edge_feat_dim=self.edge_feat_dim)
        
        #change to doing convolutions instead of points
        self.fiber_in_down_gcn   =  self.fiber_out_topkpool
        self.fiber_out_down_gcn  = Fiber({0: self.feat0*self.mult*self.mult, 1: self.feat1*self.mult})

        self.down_gcn = ConvSE3(fiber_in  = self.fiber_in_down_gcn,
                           fiber_out = self.fiber_out_down_gcn,
                           fiber_edge= self.fiber_edge,
                             self_interaction=True,
                             use_layer_norm=True,
                             max_degree=self.max_degree,
                             fuse_level= ConvSE3FuseLevel.NONE,
                             low_memory= True)
        
        
        self.fiber_in_down_gcnp = self.fiber_out_down_gcn
        #probably rename latent
        self.latent_size = self.feat0*self.mult*self.mult
        self.fiber_latent = Fiber({0: self.latent_size})

        self.down_gcn2pool = ConvSE3(fiber_in=self.fiber_in_down_gcnp,
                                     fiber_out=self.fiber_latent,
                                     fiber_edge=self.fiber_edge,
                                     self_interaction=True,
                                     use_layer_norm=True,
                                     max_degree=self.max_degree,
                                     fuse_level= ConvSE3FuseLevel.NONE,
                                     low_memory= True)
        
        self.global_pool = GPooling(pool=self.pool_type, feat_type=0)

        self.latent_unpool_layer = Latent_Unpool(fiber_in = self.fiber_latent, fiber_add = self.fiber_out_down_gcn, 
                                            knodes = self.k)

        self.up_gcn = ConvSE3(fiber_in=self.fiber_out_down_gcn,
                             fiber_out=self.fiber_out_down_gcn,
                             fiber_edge=self.fiber_edge,
                             self_interaction=True,
                             use_layer_norm=True,
                             max_degree=self.max_degree,
                             fuse_level= ConvSE3FuseLevel.NONE,
                             low_memory= True)
        
        self.unpool_layer = Unpool_Layer(fiber_in=self.fiber_out_down_gcn, fiber_add=self.fiber_out_down_ca)
        
        self.fiber_in_up_gcn_mp = self.unpool_layer.fiber_out
        self.fiber_hidden_up_mp= self.fiber_hidden_down_ca2mp
        self.fiber_out_up_gcn_mp = self.fiber_out_down_mp_out

        self.up_gcn_mp = SE3Transformer(num_layers = num_layers,
                        fiber_in=self.fiber_in_up_gcn_mp,
                        fiber_hidden= self.fiber_hidden_up_mp, 
                        fiber_out=self.fiber_out_up_gcn_mp,
                        num_heads = self.num_heads,
                        channels_div = self.channels_div,
                        fiber_edge=self.fiber_edge,
                        low_memory=True,
                        tensor_cores=False)
        
        self.unpool_layer_off_mp = Unpool_Layer(fiber_in=self.fiber_out_down_mp_out, fiber_add=self.fiber_out_down_mp_out)

        self.fiber_in_up_off_mp = self.fiber_out_up_gcn_mp
        self.fiber_hidden_up_off_mp = self.fiber_hidden_up_mp
        self.fiber_out_up_off_mp = self.fiber_out_down_ca 
        
        #uses reverse graph to move mp off 
        
        self.up_off_mp = SE3Transformer(num_layers = self.num_layers,
                        fiber_in=self.fiber_in_up_off_mp,
                        fiber_hidden= self.fiber_hidden_up_off_mp, 
                        fiber_out=self.fiber_out_up_off_mp,
                        num_heads = self.num_heads,
                        channels_div = self.channels_div,
                        fiber_edge=self.fiber_edge,
                        low_memory=True,
                        tensor_cores=False)
        
        self.pre_linear = Fiber({1:36})
        
        #concat_t, plus one on input fiber, run concat_t method on forward
        
        self.up_ca = SE3Transformer(num_layers = self.num_layers_ca,
                                    fiber_in=self.fiber_out_down_ca+self.t_fiber,
                                    fiber_hidden= self.fiber_hidden_down_ca, 
                                    fiber_out=self.pre_linear,
                                    num_heads = self.num_heads,
                                    channels_div = self.channels_div,
                                    fiber_edge= self.fiber_edge,
                                    low_memory=True,
                                    tensor_cores=False)
        
        self.fiber_out = fiber_out
        
        self.linear = LinearSE3(fiber_in=self.pre_linear,
                                fiber_out=fiber_out)
        
        if self.zero_lin:
            self.zero_linear()
        self.act = nn.SiLU()
        
    def zero_linear(self):
        nn.init.zeros_(self.linear.weights['1'])
        
    def concat_mp_feats(self, ca_feats_in, mp_feats):

        nf0_c = ca_feats_in['0'].shape[-2]
        nf1_c = ca_feats_in['1'].shape[-2]

        out0_cat_shape = (self.B,self.ca_nodes,-1,1)
        mp0_cat_shape  = (self.B,self.mp_nodes,-1,1)
        out1_cat_shape = (self.B,self.ca_nodes,-1,3)
        mp1_cat_shape  = (self.B,self.mp_nodes,-1,3)

        nf_c = {} #nf_cat
        nf_c['0'] = torch.cat((ca_feats_in['0'].reshape(out0_cat_shape), 
                                 mp_feats['0'].reshape(mp0_cat_shape)[:,-(self.mp_nodes-self.ca_nodes):,:,:]),
                              axis=1).reshape((-1,nf0_c,1))

        nf_c['1'] = torch.cat((ca_feats_in['1'].reshape(out1_cat_shape), 
                                 mp_feats['1'].reshape(mp1_cat_shape)[:,-(self.mp_nodes-self.ca_nodes):,:,:]),
                              axis=1).reshape((-1,nf1_c,3))

        return nf_c
        
    def pull_out_mp_feats(self, ca_mp_feats):

        nf0_c = ca_mp_feats['0'].shape[1]
        nf1_c = ca_mp_feats['1'].shape[1]

        nf_mp_ = {}
        #select just mp nodes to move on, the other nodes don't connect but mainting self connections
        nf_mp_['0'] = ca_mp_feats['0'].reshape(self.B,self.mp_nodes,
                                               nf0_c,1)[:,-(self.mp_nodes-self.ca_nodes):,...].reshape((-1,nf0_c,1))
        nf_mp_['1'] = ca_mp_feats['1'].reshape(self.B,self.mp_nodes,
                                               nf1_c,3)[:,-(self.mp_nodes-self.ca_nodes):,...].reshape((-1,nf1_c,3))

        return nf_mp_
    
    def concat_t(self, feats_in, embedded_t, use_deg1=True):
        """Concatenate T to first position of each tensor. Pad Zeros left for degree 1."""
        feats_out = {}
        key = next(iter(feats_in.keys()))
        shape_tuple = (self.B,-1)+feats_in[key].shape[1:]
        batch_shape = feats_in[key].reshape(shape_tuple).shape
        L = batch_shape[1] #can be ca, ca+mp, mp, k nodes long

        if '0' in feats_in.keys():
            feats_out['0'] = torch.concat((embedded_t['0'][:,None,:,None].repeat(1,L,1,1), 
                                           feats_in['0'].reshape((self.B,L,-1,1))),
                                          axis=2).reshape((self.B*L,-1,1))
        if '1' in feats_in.keys():
            
            if use_deg1:
                feats_out['1'] = torch.multiply(feats_in['1'],
                                                embedded_t['1'][:,None,:,None].repeat(1,L,1,1).reshape((self.B*L,-1,1)))
            else:
                feats_out['1'] = feats_in['1']

        return feats_out
        
    def forward(self, input_tuple, batched_t):

        b_graph, nf, ef, b_graph_mp, nf_mp, ef_mp, b_graph_mps, nf_mps, ef_mps, b_graph_mpRev = input_tuple
        #assumes equal node numbers in g raphs
        self.ca_nodes = int(b_graph.batch_num_nodes()[0])
        self.mp_nodes = int(b_graph_mp.batch_num_nodes()[0]) #ca+mp nodes number

        #SE3 Attention Transformer, c_alpha
        embed_t = self.embed_t(batched_t)
        t_nf = self.concat_t(nf, embed_t, use_deg1=self.use_tdeg1) #concat_t on
        nf_ca_down_out = self.down_ca(b_graph, t_nf, ef)

        #concatenate on midpoints feats
        
        nf_down_cat_mp = self.concat_mp_feats(nf_ca_down_out, nf_mp)

        #pool from ca onto selected midpoints via SE3 Attention transformer
        #edges from ca to mp only (ca nodes zero after this)
        t_nf_down_cat_mp = self.concat_t(nf_down_cat_mp, embed_t,use_deg1=self.use_tdeg1) #concat_t on
        nf_down_ca2mp_out = self.down_ca2mp(b_graph_mp, t_nf_down_cat_mp, ef_mp)

        #remove ca node feats from tensor 
        nf_mp_out = self.pull_out_mp_feats(nf_down_ca2mp_out)

        t_nf_mp_out = self.concat_t(nf_mp_out, embed_t, use_deg1=self.use_tdeg1) #concat_t on
        node_feats_tk, topk_feats, topk_indx = self.mp_topk(b_graph_mps, t_nf_mp_out, ef_mps)

        #make new basis for small graph of k selected midpoints
        edge_feats_out, basis_out, new_pos = prep_for_gcn(self.gsmall, b_graph_mps.ndata['pos'], self.ef_small,
                                                          topk_indx,
                                                          max_degree=self.max_degree, comp_grad=True)

        down_gcn_out = self.down_gcn(topk_feats, edge_feats_out, self.gsmall,  basis_out)

        down_gcnpool_out = self.down_gcn2pool(down_gcn_out, edge_feats_out, self.gsmall,  basis_out)

        pooled_tensor = self.global_pool(down_gcnpool_out,self.gsmall)
        pooled = {'0':pooled_tensor}
        #----------------------------------------- end of down section
        lat_unp = self.latent_unpool_layer(pooled,down_gcn_out)

        up_gcn_out = self.up_gcn(lat_unp, edge_feats_out, self.gsmall,  basis_out)

        k_to_mp  = self.unpool_layer(up_gcn_out,node_feats_tk,topk_indx)

        up_mp_gcn_out = self.up_gcn_mp(b_graph_mps, k_to_mp, ef_mps)
        
        off_mp_add = {}
        for k,v in up_mp_gcn_out.items():
            off_mp_add[k] = torch.add(up_mp_gcn_out[k],nf_mp_out[k])

        #midpoints node indices for unpool layer
        mp_node_indx = torch.arange(self.ca_nodes,self.mp_nodes, device=self.device)
        mp_idx = mp_node_indx[None,...].repeat_interleave(self.B,0)
        mp_idx =((torch.arange(self.B,device=self.device)*(self.mp_nodes)).reshape((-1,1))+mp_idx).reshape((-1))
        
        #during unpool, keep mp=values and ca=zeros
        zeros_mp_ca = {}
        for k,v in nf_down_cat_mp.items():
            zeros_mp_ca[k] = torch.zeros_like(v, device=self.device)


        unpoff_out = self.unpool_layer_off_mp(off_mp_add, zeros_mp_ca, mp_idx)
        
        out_up_off_mp = self.up_off_mp(b_graph_mpRev, unpoff_out, ef_mp)
        
        #select just ca nodes, mp = zeros from last convolution
        inv_mp_idx= torch.arange(0,self.ca_nodes, device=self.device)
        inv_mp_idx = inv_mp_idx[None,...].repeat_interleave(self.B,0)
        inv_mp_idx =((torch.arange(self.B,device=self.device)*(self.mp_nodes)).reshape((-1,1))
                     +inv_mp_idx).reshape((-1))

        node_final_ca = {}
        for key in out_up_off_mp.keys():
            node_final_ca[key] = torch.add(out_up_off_mp[key][inv_mp_idx,...],nf_ca_down_out[key])

        #return updates 
        t_node_final_ca = self.concat_t(node_final_ca, embed_t, use_deg1=self.use_tdeg1) #concat_t on
        
        return self.linear(self.up_ca(b_graph, t_node_final_ca, ef))