import torch
import numpy as np
import util.npose_util as nu
import os
import dgl
from dgl import backend as F
from torch.utils.data import random_split, DataLoader, Dataset
from typing import Dict
from torch import Tensor
from dgl import DGLGraph
from se3_transformer.runtime.utils import to_cuda
import pandas as pd
import util.framediff_utils as du
from openfold.np import residue_constants
import tree
import random
import logging
#from se3_diffuse import rigid_utils as ru

#Globals from npose for making pdb files
#imported
N_CA_dist = torch.tensor(1.458) #update this when imported
C_CA_dist = torch.tensor(1.523)

if ( hasattr(os, 'ATOM_NAMES') ):
    assert( hasattr(os, 'PDB_ORDER') )

    ATOM_NAMES = os.ATOM_NAMES
    PDB_ORDER = os.PDB_ORDER
else:
    ATOM_NAMES=['N', 'CA', 'CB', 'C', 'O']
    PDB_ORDER = ['N', 'CA', 'C', 'O', 'CB']

_byte_atom_names = []
_atom_names = []
for i, atom_name in enumerate(ATOM_NAMES):
    long_name = " " + atom_name + "       "
    _atom_names.append(long_name[:4])
    _byte_atom_names.append(atom_name.encode())

    globals()[atom_name] = i

R = len(ATOM_NAMES)

if ( "N" not in globals() ):
    N = -1
if ( "C" not in globals() ):
    C = -1
if ( "CB" not in globals() ):
    CB = -1


_pdb_order = []
for name in PDB_ORDER:
    _pdb_order.append( ATOM_NAMES.index(name) )


#-------------------------Helper functions 
def define_graph_edges(n_nodes):
    #input number of nodes: n_nodes
    #connected backbone = 1, unconnected = 0

    con_v1 = np.arange(n_nodes-1) #vertex 1 of edges in chronological order
    con_v2 = np.arange(1,n_nodes) #vertex 2 of edges in chronological order

    ind = con_v1*(n_nodes-1)+con_v2-1 #account for removed self connections (-1)


    #unconnected backbone

    nodes = np.arange(n_nodes)
    v1 = np.repeat(nodes,n_nodes-1) #starting vertices, same number repeated for each edge

    start_v2 = np.repeat(np.arange(n_nodes)[None,:],n_nodes,axis=0)
    diag_ind = np.diag_indices(n_nodes)
    start_v2[diag_ind] = -1 #diagonal of matrix is self connections which we remove (self connections are managed by SE3 Conv channels)
    v2 = start_v2[start_v2>-0.5] #remove diagonal and flatten

    edge_data = torch.zeros(len(v2))
    edge_data[ind] = 1
    
    return v1,v2,edge_data, ind

def make_pe_encoding(n_nodes=65, embed_dim = 12, scale = 40, cast_type=torch.float32, print_out=False):
    #positional encoding of node, scale dimension optimized for 65
    i_array = np.arange(1,(embed_dim/2)+1)
    wk = (1/(scale**(i_array*2/embed_dim)))
    t_array = np.arange(n_nodes)
    si = torch.tensor(np.sin(wk*t_array.reshape((-1,1))))
    ci = torch.tensor(np.cos(wk*t_array.reshape((-1,1))))
    pe = torch.stack((si,ci),axis=2).reshape(t_array.shape[0],embed_dim).type(cast_type)
    
    if print_out == True:
        for x in range(int(n_nodes/12)):
            print(np.round(pe[x],1))
    
    return pe

def _get_relative_pos(graph_in: dgl.DGLGraph) -> torch.Tensor:
    x = graph_in.ndata['pos']
    src, dst = graph_in.edges()
    rel_pos = x[dst] - x[src]
    return rel_pos

def torch_normalize(v, eps=1e-6):
    """Normalize vector in last axis"""
    norm = torch.linalg.vector_norm(v, dim=len(v.shape)-1)+eps
    return v / norm[...,None]

def normalize(v):
    """Normalize vector in last axis"""
    norm = np.linalg.norm(v,axis=len(v.shape)-1)
    norm[norm == 0] = 1
    return v / norm[...,None]

def gudiff_parse_chain_feats(chain_feats, scale_factor=10., cast_type=torch.float32):
    ca_idx = residue_constants.atom_order['CA']
    n_idx = residue_constants.atom_order['N']
    c_idx = residue_constants.atom_order['C']
    chain_feats['bb_mask'] = chain_feats['atom_mask'][:, ca_idx]
    
    bb_pos = chain_feats['atom_positions'][:, ca_idx]/scale_factor #scale factor mod
    bb_center = np.sum(bb_pos, axis=0) / (np.sum(chain_feats['bb_mask']) + 1e-5)
    centered_pos = chain_feats['atom_positions'] - bb_center[None, None, :]
    
    
    coordinates = centered_pos/scale_factor
    #unsqueeze to stack together later
    N_CA_vec = torch.tensor(coordinates[:,n_idx] - coordinates[:,ca_idx], dtype=cast_type)/scale_factor
    C_CA_vec = torch.tensor(coordinates[:,c_idx] - coordinates[:,ca_idx], dtype=cast_type)/scale_factor
        
    N_CA_vec = torch_normalize(N_CA_vec)#.unsqueeze(2) #do the unsqueeze later
    C_CA_vec = torch_normalize(C_CA_vec)#.unsqueeze(2)

    scaled_pos = centered_pos / scale_factor
    chain_feats['atom_positions'] = scaled_pos * chain_feats['atom_mask'][..., None]
    
    chain_feats['CA'] = torch.tensor(coordinates[:,ca_idx],dtype=cast_type)
    chain_feats['N_CA_vec'] = N_CA_vec
    chain_feats['C_CA_vec'] = C_CA_vec
    return chain_feats


class smallPDBDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            diffuser,
            conf,
            filter_dict=True,
            maxlen=None,
            is_training=True,
            input_t=None,
            t_range=None,
            swap_metadir=False,
        ):
        #self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self.meta_data_path =  conf['meta_data_path']
        self.swap_metadir=swap_metadir
        self.conf = conf
        self._init_metadata(filter_dict=filter_dict,maxlen=maxlen) #includes create split that saves self.csv
        self._diffuser = diffuser
        self.input_t = input_t
        self.t_range = t_range
        self.verbose = True

        
    @property
    def is_training(self):
        return self._is_training

    @property
    def diffuser(self):
        return self._diffuser

    def _init_metadata(self, filter_dict=True, maxlen=None):
        """Initialize metadata."""
        
        #meta_data_path = '/mnt/h/datasets/p200/metadata.csv'
        pdb_csv = pd.read_csv(self.meta_data_path)




        
        if self.swap_metadir: #file location is hard code into meta data, this will take the directory location of the meta_data_csv and apply it to the filenames
            dir_path = os.path.dirname(os.path.realpath(self.meta_data_path))
            pdb_csv['processed_path'] = pdb_csv['processed_path'].apply(lambda x: os.path.join(dir_path, os.path.basename((os.path.dirname(x))) ,os.path.basename(x)))


        
        if filter_dict:
            filter_conf = {'allowed_oligomer': ['monomeric'],
                           'max_loop_percent': 0.5}
            pdb_csv = pdb_csv[pdb_csv.oligomeric_detail.isin(filter_conf['allowed_oligomer'])]
            pdb_csv = pdb_csv[pdb_csv.coil_percent < filter_conf['max_loop_percent']]
            pdb_csv = pdb_csv[pdb_csv.modeled_seq_len > self.conf['min_len']]
            pdb_csv = pdb_csv[pdb_csv.modeled_seq_len < self.conf['max_len']]
            pdb_csv = pdb_csv.sort_values('modeled_seq_len', ascending=False)
            
        if maxlen >1:
            pdb_csv = pdb_csv.sample(n=maxlen)
        #self._create_split(pdb_csv)
        self.csv = pdb_csv

    
    def _create_split(self, pdb_csv):
        # Training or validation specific logic.
        #if self.is_training:
        self.csv = pdb_csv
        #self._log.info(
        #    f'Training: {len(self.csv)} examples')
#         else:
#             all_lengths = np.sort(pdb_csv.modeled_seq_len.unique())
#             length_indices = (len(all_lengths) - 1) * np.linspace(
#                 0.0, 1.0, self._data_conf.num_eval_lengths)
#             length_indices = length_indices.astype(int)
            
#             if self._simple:
#                 eval_lengths = np.array([65]).astype(int)
#             else:
#                 eval_lengths = all_lengths[length_indices]
                
#             eval_csv = pdb_csv[pdb_csv.modeled_seq_len.isin(eval_lengths)]
#             # Fix a random seed to get the same split each time.
#             eval_csv = eval_csv.groupby('modeled_seq_len').sample(
#                 self._data_conf.samples_per_eval_length, replace=True, random_state=123)
#             eval_csv = eval_csv.sort_values('modeled_seq_len', ascending=False)
#             self.csv = eval_csv
#             self._log.info(
#                 f'Validation: {len(self.csv)} examples with lengths {eval_lengths}')
    # cache make the same sample in same batch 
    #@fn.lru_cache(maxsize=100)
    def _process_csv_row(self, processed_file_path, index=0):
        
        processed_feats = du.read_pkl(processed_file_path)
        chain_feats = gudiff_parse_chain_feats(processed_feats,scale_factor=10.)
        
        # Only take modeled residues.
        modeled_idx = processed_feats['modeled_idx']
        min_idx = np.min(modeled_idx)
        max_idx = np.max(modeled_idx)
        del processed_feats['modeled_idx']
        processed_feats = tree.map_structure(
            lambda x: x[min_idx:(max_idx+1)], processed_feats)
        chain_feats = tree.map_structure(
            lambda x: x[min_idx:(max_idx+1)], chain_feats)
        

        # Run through OpenFold data transforms.
        # Re-number residue indices for each chain such that it starts from 1.
        # Randomize chain indices.
        chain_idx = processed_feats["chain_index"]
        res_idx = processed_feats['residue_index']
        new_res_idx = np.zeros_like(res_idx)
        new_chain_idx = np.zeros_like(res_idx)
        all_chain_idx = np.unique(chain_idx).tolist()
        shuffled_chain_idx = np.array(
            random.sample(all_chain_idx, len(all_chain_idx))) - np.min(all_chain_idx) + 1
        for i,chain_id in enumerate(all_chain_idx):
            chain_mask = (chain_idx == chain_id).astype(int)
            chain_min_idx = np.min(res_idx + (1 - chain_mask) * 1e3).astype(int)
            new_res_idx = new_res_idx + (res_idx - chain_min_idx + 1) * chain_mask

            # Shuffle chain_index
            replacement_chain_id = shuffled_chain_idx[i]
            new_chain_idx = new_chain_idx + replacement_chain_id * chain_mask

        # To speed up processing, only take necessary features
        final_feats = {
            'chain_idx': new_chain_idx,
            'residue_index': processed_feats['residue_index'],
            'res_mask': processed_feats['bb_mask'],
            'CA':   chain_feats['CA'],
            'N_CA': chain_feats['N_CA_vec'], #when unsqueeze? later maybe take time to change this behavior
            'C_CA': chain_feats['C_CA_vec'],
            'file_path_index' : index
        }
        return final_feats
    
    def get_specific_t(self, idx, t):
        example_idx = idx
        csv_row = self.csv.iloc[example_idx]
        if 'pdb_name' in csv_row:
            pdb_name = csv_row['pdb_name']
        elif 'chain_name' in csv_row:
            pdb_name = csv_row['chain_name']
        else:
            raise ValueError('Need chain identifier.')
            
        processed_file_path = csv_row['processed_path']
        chain_feats = self._process_csv_row(processed_file_path,index=idx)

        bb_noised =  self._diffuser.forward(chain_feats, t=t)
        chain_feats.update(bb_noised)

        # Convert all features to tensors.
        final_feats = tree.map_structure(
                    lambda x: x if torch.is_tensor(x) else torch.tensor(x), chain_feats)
        
        return final_feats
        
        
    def __getitem__(self, idx):
        example_idx = idx
        csv_row = self.csv.iloc[example_idx]
        if self.verbose:
            print(csv_row['pdb_name'], csv_row['modeled_seq_len'], 'cluster: ' , csv_row['cluster'] )
        if 'pdb_name' in csv_row:
            pdb_name = csv_row['pdb_name']
        elif 'chain_name' in csv_row:
            pdb_name = csv_row['chain_name']
        else:
            raise ValueError('Need chain identifier.')
        processed_file_path = csv_row['processed_path']
        chain_feats = self._process_csv_row(processed_file_path, index=idx)

        # Use a fixed seed for evaluation.
#         if self.is_training:
        rng = np.random.default_rng(None)
#         else:
#             rng = np.random.default_rng(idx)

        # Sample t and diffuse.
#         if self.is_training:
        if self.input_t is None and self.t_range is None:
            t = rng.uniform(1e-3, 1.0)
        elif self.t_range is not None:
            t = rng.uniform(self.t_range[0], self.t_range[1])
        else:
            t = self.input_t
        bb_noised =  self._diffuser.forward(chain_feats, t=t)
#         else:
#             t = 1.0
#             diff_feats_t = self.diffuser.sample_ref(
#                 n_samples=gt_bb_rigid.shape[0],
#                 impute=gt_bb_rigid,
#                 diffuse_mask=None,
#                 as_tensor_7=True,
#             )
        chain_feats.update(bb_noised)

        # Convert all features to tensors.
        final_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), chain_feats)
        #final_feats = du.pad_feats(final_feats, csv_row['modeled_seq_len'])
        #if self.is_training:
#         else:
#             return final_feats, pdb_name
        
        return final_feats
    
    def __len__(self):
        return len(self.csv)


class TrainSampler(torch.utils.data.Sampler):

    def __init__(self, batch_size, dataset, data_conf,
                 sample_mode='length_batch'):
        
        self.dataset = dataset
        self._data_csv = dataset.csv
        self._dataset_indices = list(range(len(self._data_csv)))
        self._data_csv['index'] = self._dataset_indices
        self._batch_size = batch_size
        self.epoch = 0
        self._sample_mode = sample_mode
        self.sampler_len = len(self._dataset_indices) * self._batch_size
        self.min_t = 1e-3
        self._log = logging.getLogger(__name__)
        self._data_conf = data_conf
        #self._dataset = dataset
        #self._data_csv = self._dataset.csv
        if self._sample_mode in ['cluster_length_batch', 'cluster_time_batch']:
            self._pdb_to_cluster = self._read_clusters()
            self._max_cluster = max(self._pdb_to_cluster.values())
            self._log.info(f'Read {self._max_cluster} clusters.')
            self._missing_pdbs = 0
            def cluster_lookup(pdb):
                pdb = pdb.upper()
                if pdb not in self._pdb_to_cluster:
                    self._pdb_to_cluster[pdb] = self._max_cluster + 1
                    self._max_cluster += 1
                    self._missing_pdbs += 1
                return self._pdb_to_cluster[pdb]
            self._data_csv['cluster'] = self._data_csv['pdb_name'].map(cluster_lookup)
            num_clusters = len(set(self._data_csv['cluster']))
            self.sampler_len = num_clusters * self._batch_size
            self._log.info(
                f'Training on {num_clusters} clusters. PDBs without clusters: {self._missing_pdbs}'
    )

    def _read_clusters(self):
        pdb_to_cluster = {}
        with open(self._data_conf['cluster_path'], "r") as f:
            for i,line in enumerate(f):
                for chain in line.split(' '):
                    pdb = chain.split('_')[0]
                    pdb_to_cluster[pdb.upper()] = i
        return pdb_to_cluster

    def __iter__(self):


        if self._sample_mode == 'length_batch':
            pdb_csv_batched =  self._data_csv.copy()
            sampled_order = pd.DataFrame(columns= self._data_csv.columns)

            while len(pdb_csv_batched)>1:
                pdb_csv_batched = pdb_csv_batched.loc[~pdb_csv_batched['index'].isin(sampled_order['index'])].copy()
                sampled_order_part = pdb_csv_batched.groupby('modeled_seq_len').sample(
                            4, replace=True, random_state=0) #one batch per length
                sampled_order = pd.concat((sampled_order,sampled_order_part))
            # Each batch contains multiple proteins of the same length., only samples each modeled seq len at max batch each epoch
            # sampled_order = self._data_csv.groupby('modeled_seq_len').sample(
            #     self._batch_size, replace=True, random_state=self.epoch) #one batch per length
            return iter(sampled_order['index'].tolist())
        elif self._sample_mode == 'single_length':
            rand_index = self._data_csv['index'].to_numpy()
            np.random.shuffle(rand_index)
            num_batches = int(rand_index.shape[0]/self._batch_size)
            rand_index = rand_index[:(num_batches*self._batch_size)] #drop last batch
            return iter(rand_index)
        elif self._sample_mode == 'cluster_length_batch':
            # Each batch contains multiple clusters of the same length.
            sampled_clusters = self._data_csv.groupby('cluster').sample(
                1, random_state=1001+self.epoch)
            sampled_order = sampled_clusters.groupby('modeled_seq_len').sample(
                self._batch_size, replace=True, random_state=self.epoch+1001)
            n = self._batch_size
            so = sampled_order 
            # Calculate the number of blocks
            num_blocks = len(so) // n

            # Create an array of block indices
            block_indices = np.arange(num_blocks)

            # Shuffle the block indices
            np.random.shuffle(block_indices)
            # Create a mask based on the shuffled block indices
            mask = np.hstack([np.arange(i*n, (i+1)*n) for i in block_indices])
            # Apply the mask to the DataFrame to reorder it in shuffled blocks
            shuffled_df = so.iloc[mask].reset_index(drop=True)


            return iter(shuffled_df['index'].tolist())
        # elif self._sample_mode == 'cluster_time_batch':
        #     # Each batch contains multiple time steps of a protein from a cluster.
        #     sampled_clusters = self._data_csv.groupby('cluster').sample(
        #         1, random_state=self.epoch)
        #     dataset_indices = sampled_clusters['index'].tolist()
        #     repeated_indices = np.repeat(dataset_indices, self._batch_size)
        #     return iter(repeated_indices.tolist())
        else:
            raise ValueError(f'Invalid sample mode: {self._sample_mode}')
    
    def generate_batch_indices(self, protein_length=128):

        if self._sample_mode == 'length_batch':
            # Each batch contains multiple proteins of the same length.
            sel_length = self._data_csv.loc[self._data_csv['modeled_seq_len']==protein_length]
            if len(sel_length)<1:
                sel_length = self._data_csv.loc[self._data_csv['modeled_seq_len']==self._data_csv['modeled_seq_len'].min()]

            sampled_order = sel_length.groupby('modeled_seq_len').sample(
                self._batch_size, replace=True, random_state=self.epoch) #one batch per length
            return sampled_order['index'].tolist()[:self._batch_size]
        
        elif self._sample_mode == 'single_length':
            rand_index = self._data_csv['index'].to_numpy()
            np.random.shuffle(rand_index)
            rand_index = rand_index[:(self._batch_size)] #drop last batch
            return rand_index
        
        elif self._sample_mode == 'cluster_length_batch':
            # Each batch contains multiple clusters of the same length.
            sampled_clusters = self._data_csv.groupby('cluster').sample(
                1, random_state=0)
            sampled_order = sampled_clusters.groupby('modeled_seq_len').sample(
                self._batch_size, replace=True, random_state=self.epoch)
              # Each batch contains multiple clusters of the same length.
            n = self._batch_size
            so = sampled_order 
            # Calculate the number of blocks
            num_blocks = len(so) // n

            # Create an array of block indices
            block_indices = np.arange(num_blocks)

            # Shuffle the block indices
            np.random.shuffle(block_indices)
            # Create a mask based on the shuffled block indices
            mask = np.hstack([np.arange(i*n, (i+1)*n) for i in block_indices])
            # Apply the mask to the DataFrame to reorder it in shuffled blocks
            shuffled_df = so.iloc[mask].reset_index(drop=True)

            return shuffled_df['index'].tolist()[:self._batch_size]
        # elif self._sample_mode == 'cluster_time_batch':
        #     # Each batch contains multiple time steps of a protein from a cluster.
        #     sampled_clusters = self._data_csv.groupby('cluster').sample(
        #         1, random_state=self.epoch)
        #     dataset_indices = sampled_clusters['index'].tolist()
        #     repeated_indices = np.repeat(dataset_indices, self._batch_size)
        #     return iter(repeated_indices.tolist())
        else:
            raise ValueError(f'Invalid sample mode: {self._sample_mode}')



            
    def __len__(self):
        return len(self.csv)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.sampler_len
    

    
    
class Make_KNN_MP_Graphs():
    
    #8 long positional encoding
    NODE_FEATURE_DIM_0 = 12
    EDGE_FEATURE_DIM = 1 # 0 or 1 primary seq connection or not
    
    def __init__(self, mp_stride=4, coord_div=10, cast_type=torch.float32, channels_start=32,
                       ndf1=6, ndf0=32,cuda=True):
        
        self.KNN = 30
        self.mp_stride = mp_stride
        self.cast_type = cast_type
        self.channels_start = channels_start
        
        self.cuda = cuda
        self.ndf1 = ndf1 #awkard adding of nodes features to mpGraph
        self.ndf0 = ndf0
        
        assert self.ndf1%3==0
        self.ndf1_div = int(self.ndf1/3)
        
    def create_and_batch(self, bb_dict, n_nodes):
        
        graphList = []
        mpGraphList = []
        mpRevGraphList = []
        mpSelfGraphList = []
        
        pe = make_pe_encoding(n_nodes=bb_dict['CA'].shape[1])
        
        for j, caXYZ in enumerate(bb_dict['CA']):
            graph = dgl.knn_graph(caXYZ, self.KNN)
            graph.ndata['pe'] = pe.to(caXYZ.device)
            graph.ndata['pos'] = caXYZ
            graph.ndata['bb_ori'] = torch.cat((bb_dict['N_CA'][j],  bb_dict['C_CA'][j]),axis=1)
            
            #define covalent connections
            esrc, edst = graph.edges()
            graph.edata['con'] = (torch.abs(esrc-edst)==1).type(self.cast_type).reshape((-1,1))
            
            mp_list = torch.zeros((len(list(range(0, n_nodes, self.mp_stride))),caXYZ.shape[1]),device=caXYZ.device)
            
            new_src = torch.tensor([],dtype=torch.int,device=caXYZ.device)
            new_dst = torch.tensor([],dtype=torch.int,device=caXYZ.device)
            
            new_src_rev = torch.tensor([], dtype=torch.int,device=caXYZ.device)
            new_dst_rev = torch.tensor([], dtype=torch.int,device=caXYZ.device)
           
            i=0#mp list counter
            for x in range(0, n_nodes, self.mp_stride):
                src, dst = graph.in_edges(x) #dst repeats x
                n_tot = torch.cat((torch.tensor(x,device=caXYZ.device).unsqueeze(0),src)) #add x to node list
                mp_list[i] = caXYZ[n_tot].sum(axis=0)/n_tot.shape[0]
                mp_node = i + graph.num_nodes() #add midpoints nodes at end of graph
                #define edges between midpoint nodes and nodes defining midpoint for midpointGraph
                
                new_src = torch.cat((new_src,n_tot))
                new_dst = torch.cat((new_dst,
                                     (torch.tensor(mp_node,device=caXYZ.device).unsqueeze(0).repeat(n_tot.shape[0]))))
                #and reverse graph for coming off
                new_src_rev = torch.cat((new_src_rev,
                                         (torch.tensor(mp_node,device=caXYZ.device).unsqueeze(0).repeat(n_tot.shape[0]))))
                new_dst_rev = torch.cat((new_dst_rev,n_tot))
                
                i+=1
                
            mpGraph = dgl.graph((new_src,new_dst))
            mpGraph.ndata['pos'] = torch.cat((caXYZ,mp_list),axis=0).type(self.cast_type)
            mp_node_indx = torch.arange(0, n_nodes, self.mp_stride).type(torch.int)
            #match output shape of first transformer
            pe_mp = torch.cat(
                              (pe.to(caXYZ.device),
                              torch.zeros( (pe.shape[0],
                                            self.ndf0-pe.shape[1]),
                                                device=caXYZ.device)) # had channels start_0, originally
                                                                     ,axis=1)
            mpGraph.ndata['pe'] = torch.cat((pe_mp,pe_mp[mp_node_indx]))
            mpGraph.edata['con'] = torch.zeros((mpGraph.num_edges(),1),device=caXYZ.device)
            
            mpGraph_rev = dgl.graph((new_src_rev,new_dst_rev))
            mpGraph_rev.ndata['pos'] = torch.cat((caXYZ,mp_list),axis=0).type(self.cast_type)
            mpGraph_rev.ndata['pe'] = torch.cat((pe_mp,pe_mp[mp_node_indx]))
            mpGraph_rev.edata['con'] = torch.zeros((mpGraph_rev.num_edges(),1),device=caXYZ.device)
            
            #make graph for self interaction of midpoints
            v1,v2,edge_data, ind = define_graph_edges(len(mp_list))
            mpSelfGraph = dgl.graph((v1,v2))
            mpSelfGraph.edata['con'] = edge_data.reshape((-1,1))
            mpSelfGraph.ndata['pe'] = pe[mp_node_indx] #not really needed
            mpSelfGraph = mpSelfGraph.to(caXYZ.device)
            mpSelfGraph.ndata['pos'] = mp_list.type(self.cast_type)
            
            
            mpSelfGraphList.append(mpSelfGraph) 
            mpGraphList.append(mpGraph)
            mpRevGraphList.append(mpGraph_rev)
            graphList.append(graph)
        
        return dgl.batch(graphList), dgl.batch(mpGraphList), dgl.batch(mpSelfGraphList), dgl.batch(mpRevGraphList)
    
    def prep_for_network(self, bb_dict, cuda=True):
    
        n_nodes = bb_dict['CA'].shape[1]
        batched_graph, batched_mpgraph, batched_mpself_graph, batched_mpRevgraph =  self.create_and_batch(bb_dict, n_nodes)
        
        edge_feats        =    {'0':   batched_graph.edata['con'][:, :self.EDGE_FEATURE_DIM, None]}
        edge_feats_mp     = {'0': batched_mpgraph.edata['con'][:, :self.EDGE_FEATURE_DIM, None]} #def all zero now
        edge_feats_mpself = {'0': batched_mpself_graph.edata['con'][:, :self.EDGE_FEATURE_DIM, None]}
#         edge_feats_mp     = {'0': batched_mpRevgraph.edata['con'][:, :self.EDGE_FEATURE_DIM, None]}
        batched_graph.edata['rel_pos']   = _get_relative_pos(batched_graph)
        batched_mpgraph.edata['rel_pos'] = _get_relative_pos(batched_mpgraph)
        batched_mpself_graph.edata['rel_pos'] = _get_relative_pos(batched_mpself_graph)
        batched_mpRevgraph.edata['rel_pos'] = _get_relative_pos(batched_mpRevgraph)
        # get node features
        
        node_feats =         {'0': batched_graph.ndata['pe'][:, :self.NODE_FEATURE_DIM_0, None],
                              '1': batched_graph.ndata['bb_ori'][:,:self.ndf1_div, :3]}
        node_feats_mp =      {'0': batched_mpgraph.ndata['pe'][:, :self.ndf0, None],
                              '1': torch.ones((batched_mpgraph.num_nodes(),self.ndf1,3))}
        #unused
        node_feats_mpself =  {'0': batched_mpself_graph.ndata['pe'][:, :self.NODE_FEATURE_DIM_0, None]}
        
        if cuda:
            bg,nf,ef = to_cuda(batched_graph), to_cuda(node_feats), to_cuda(edge_feats)
            bg_mp, nf_mp, ef_mp = to_cuda(batched_mpgraph), to_cuda(node_feats_mp), to_cuda(edge_feats_mp)
            bg_mps, nf_mps, ef_mps = to_cuda(batched_mpself_graph), to_cuda(node_feats_mpself), to_cuda(edge_feats_mpself)
            bg_mpRev = to_cuda(batched_mpRevgraph)
            
            return bg,nf,ef, bg_mp, nf_mp, ef_mp, bg_mps, nf_mps, ef_mps, bg_mpRev
        
        else:
            bg,nf,ef = batched_graph, node_feats, edge_feats
            bg_mp, nf_mp, ef_mp = batched_mpgraph, node_feats_mp, edge_feats_mp
            bg_mps, nf_mps, ef_mps = batched_mpself_graph, node_feats_mpself, edge_feats_mpself
            bg_mpRev = batched_mpRevgraph
            
            return bg,nf,ef, bg_mp, nf_mp, ef_mp, bg_mps, nf_mps, ef_mps, bg_mpRev
        


def get_edge_features(graph,edge_feature_dim=1):
    return {'0': graph.edata['con'][:, :edge_feature_dim, None]}

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
    
def build_npose_from_coords(coords_in):
    """Use N, CA, C coordinates to generate O an CB atoms. """
    rot_mat_cat = np.ones(sum((coords_in.shape[:-1], (1,)), ()))
    
    coords = np.concatenate((coords_in,rot_mat_cat),axis=-1)
    
    npose = np.ones((coords_in.shape[0]*5,4)) #5 is atoms per res

    by_res = npose.reshape(-1, 5, 4)
    
    if ( "N" in ATOM_NAMES ):
        by_res[:,N,:3] = coords_in[:,0,:3]
    if ( "CA" in ATOM_NAMES ):
        by_res[:,CA,:3] = coords_in[:,1,:3]
    if ( "C" in ATOM_NAMES ):
        by_res[:,C,:3] = coords_in[:,2,:3]
    if ( "O" in ATOM_NAMES ):
        by_res[:,O,:3] = nu.build_O(npose)
    if ( "CB" in ATOM_NAMES ):
        tpose = nu.tpose_from_npose(npose)
        by_res[:,CB,:] = nu.build_CB(tpose)

    return npose

def dump_coord_pdb(coords_in, fileOut='fileOut.pdb'):
    
    npose =  build_npose_from_coords(coords_in)
    nu.dump_npdb(npose,fileOut)
        