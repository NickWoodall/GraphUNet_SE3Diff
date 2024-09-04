from data_rigid_diffuser import so3_diffuser
from data_rigid_diffuser import r3_diffuser
from data_rigid_diffuser import oneHot_diffuser
from scipy.spatial.transform import Rotation
from data_rigid_diffuser import rigid_utils as ru
import se3_diffuse.utils as du
import yaml
import torch
import numpy as np

from gudiff_model import PDBDataSet_GraphCon

#find better way to incorporate coord_scale
#needed
N_CA_dist = (PDBDataSet_GraphCon.N_CA_dist/10.)
C_CA_dist = (PDBDataSet_GraphCon.C_CA_dist/10.)

#stubs for starting relative NC_CC vecs
stub = np.array([[-1.45837285,  0 , 0],         #N
         [0., 0., 0.],                 #CA
        [0.55221403, 1.41890368, 0. ]] ) #C
nca_stub = PDBDataSet_GraphCon.torch_normalize(torch.tensor(stub[0]-stub[1]))
cca_stub = PDBDataSet_GraphCon.torch_normalize(torch.tensor(stub[2]-stub[1]))

def _extract_trans_rots(rigid: ru.Rigid):
    rot = rigid.get_rots().get_rot_mats().cpu().numpy()
    rot_shape = rot.shape
    num_rots = np.cumprod(rot_shape[:-2])[-1]
    rot = rot.reshape((num_rots, 3, 3))
    rot = Rotation.from_matrix(rot).as_rotvec().reshape(rot_shape[:-2] +(3,))
    tran = rigid.get_trans().cpu().numpy()
    return tran, rot

def rotvec_2_ncVec(rot_vec,B,L,cast = torch.float32):
    
#     rotmat = Rotation.from_rotvec(rot_ref).as_matrix()
#     nc_vec = ru.rot_vec_mul(torch.tensor(rotmat,dtype=cast),ncs.reshape((-1,3))).reshape(batch_shape)
#     cc_vec = ru.rot_vec_mul(torch.tensor(rotmat,dtype=cast),ccs.reshape((-1,3))).reshape(batch_shape)
    ncs = nca_stub[None,None,None,:].repeat(B,L,1,1)
    ccs = cca_stub[None,None,None,:].repeat(B,L,1,1)
    batch_shape =  ncs.shape
    
    rotmat=du.rotvec_to_matrix(rot_vec.reshape((-1,3)))
    nc_vec = ru.rot_vec_mul(torch.tensor(rotmat,dtype=cast),ncs.reshape((-1,3))).reshape(batch_shape)
    cc_vec = ru.rot_vec_mul(torch.tensor(rotmat,dtype=cast),ccs.reshape((-1,3))).reshape(batch_shape)
    
    return nc_vec, cc_vec

def _assemble_rigid(rotvec, trans):
    rotvec_shape = rotvec.shape
    num_rotvecs = np.cumprod(rotvec_shape[:-1])[-1]
    rotvec = rotvec.reshape((num_rotvecs, 3))
    rotmat = Rotation.from_rotvec(rotvec).as_matrix().reshape(
        rotvec_shape[:-1] + (3, 3))
    return ru.Rigid(
            rots=ru.Rotation(
                rot_mats=torch.Tensor(rotmat)),
            trans=torch.tensor(trans))

# Applies to Python-3 Standard Library
class Struct(object):
    def __init__(self, data):
        for name, value in data.items():
            setattr(self, name, self._wrap(value))

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)): 
            return type(value)([self._wrap(v) for v in value])
        else:
            return Struct(value) if isinstance(value, dict) else value

class FrameDiffNoise(torch.nn.Module):
    """Generate Diffusion Noise based on FrameDiff"""
    
    def __init__(self, config_path='data_rigid_diffuser/base.yaml', max_t = 1.0, roll=True):
        super().__init__()
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        conf = Struct(config['diffuser'])
        
        self.so3d = so3_diffuser.SO3Diffuser(conf.so3)
        self.r3d =  r3_diffuser.R3Diffuser(conf.r3)
        self.ohd = oneHot_diffuser.OHDiffuser(conf.r3)
        self.minMarker = -1e8
        self.maxMarker = 1e8
        
        self.last_gi = None
        self.mask_orig = None
        self.mask_shift = None
        self.mask_start = None
        self.mask_end = None
        
        self.roll=roll
        self.max_t = max_t
        
        self.node_type_dim = 5 #number of voting features for real or null nodes
        
    def score_scaling(self, t, useR3 = True):
        if useR3:
            score_scaling = self.r3d.score_scaling(t)
        else:
            score_scaling = self.so3d.score_scaling(t)
        return score_scaling
    
    def prep_random_shift(self, tens_in, shift=True, no_mask=False):
        """records mask for indexing to shift"""
        
        #now take bb_dict and repeat terminal ends and beginnings here
        lengths_aa = (tens_in[:,:,0]>self.minMarker).sum(axis=1)
        max_possible_starts = tens_in.shape[1] - lengths_aa
        last_gi = (lengths_aa[...,None].repeat((1,3))-1).unsqueeze(1)
        
        #randomize position of protein in nodes
        rng = np.random.default_rng()
        
        if shift and not no_mask:
            randstart = torch.tensor(rng.integers(0,max_possible_starts)) #exclusive high value, inclusive low
        else:
            randstart = torch.zeros_like(max_possible_starts)
            
        end = torch.add(randstart,lengths_aa)
        
        # # Range array for the length of columns
        r = torch.arange(tens_in.shape[1])
        
        mask_shift = (randstart[:,None] <= r) & (end[:,None] > r) #index the shifted nodes
        mask_orig = (torch.zeros((lengths_aa.shape[0],1))<=r) & (lengths_aa[:,None]>r) #index zero index values #are repeated

        mask_start = (randstart[:,None] > r)
        mask_end   = (end[:,None] <= r)

        
        self.last_gi = last_gi
        self.lengths_aa = lengths_aa
        self.mask_orig = mask_orig
        self.mask_shift = mask_shift
        self.mask_start = mask_start
        self.mask_end = mask_end
        self.randstart = randstart
        if self.roll==True:
            self.randroll = rng.integers(-tens_in.shape[1],tens_in.shape[1]) #just shift all by one number
        else:
            self.randroll = 0
        
    
    def shift_nodes(self, tens_in, shift=True):
        """Shift determines number of repeated endpoints, Roll controls relative position"""
        
        if self.mask_start is None:
            self.prep_random_shift(tens_in,shift=shift)
           
        #get first and last xyz for filling
        first_point = tens_in[:,0,...][:,None,...]
        last_point = tens_in.gather(1,self.last_gi)
        if shift:
            shifted = torch.zeros_like(tens_in)
            shifted[self.mask_shift] = tens_in[self.mask_orig]
            shifted[self.mask_start] =  first_point.repeat((1,shifted.shape[1],1))[self.mask_start]
            shifted[self.mask_end] =  last_point.repeat((1,shifted.shape[1],1))[self.mask_end]
            shifted = torch.roll(shifted,self.randroll,dims=1) #roll=0 if self.roll=False
        else:
            shifted=tens_in.clone()
        
        return shifted, first_point, last_point
    
    def get_shift_roll(self):
        return self.randstart, self.randroll
        
    def create_edge_fill(self,prot_shape,edge_dim=4,k=30,fill=2):
        """Create full set of values to use as edges will choose whether 1 or 0 during graph making"""
        edge_dim=3  #real and direct connect, real and non-direct, false and direct # no false and not direct
                                #batch     #num_nodes(max aa)
        edge_fill = torch.ones((prot_shape[0],prot_shape[1],k,edge_dim,2))#extra dimension to have value for 
                                                                       #1, 0 premade and choose once graph is made
        edge_fill[...,0]=0
        
        return edge_fill
    
    def forward_with_null(self, bb_dict, t_vec=None, node_feats=None, k=30, useR3=True ,
                cast=torch.float32, shift=True, roll=False, no_mask=False):
        ca = bb_dict['CA']
        
        if t_vec is None:
            t_vec =  np.random.uniform(size=ca.shape[0])*self.max_t
        score_scales = np.array([self.score_scaling(t, useR3) for t in t_vec])
        
        #pass this to gm_maker, mask is saved to object
        self.prep_random_shift(ca, shift=shift, no_mask=no_mask) #creates real/null mask based on inputs
        
        ###############Follow along here 
        ca, first_point, last_point = self.shift_nodes(ca, shift=shift)
        nc_vec, first_point_nc, last_point_nc = self.shift_nodes(bb_dict['N_CA'].squeeze(), shift=shift)
        cc_vec, first_point_cc, last_point_cc = self.shift_nodes(bb_dict['C_CA'].squeeze(), shift=shift)

        nc_vec =  nc_vec.reshape((-1,3))
        cc_vec = cc_vec.reshape((-1,3))
        
        bb_shifted = {'CA': ca, 'N_CA': nc_vec, 'C_CA': cc_vec}
        bb_firstp = {'CA': first_point, 'N_CA': first_point_nc, 'C_CA':   first_point_cc}
        bb_lastp = {'CA': last_point, 'N_CA': last_point_nc, 'C_CA': last_point_cc}
        
        edge_fill = self.create_edge_fill(ca.shape)
        #edges_noise = np.array([self.ohd.forward_marginal(edge_fill[i].numpy(),t)[0] for i,t in enumerate(t_vec)])
        
        #sample rotation
        rot_vec = np.array([self.so3d.sample(t, n_samples=ca.shape[1]) for t in t_vec]).reshape((-1,3))
        rotmat = Rotation.from_rotvec(rot_vec).as_matrix()
        
        batch_shape =  bb_dict['N_CA'].shape
        nc_vec_noised = ru.rot_vec_mul(torch.tensor(rotmat,dtype=cast),nc_vec).reshape(batch_shape)
        cc_vec_noised = ru.rot_vec_mul(torch.tensor(rotmat,dtype=cast),cc_vec).reshape(batch_shape)
        
        #get apply transation to CA
        ca_noised = np.array([self.r3d.forward_marginal(ca[i].numpy(),t)[0] for i,t in enumerate(t_vec)])
        ca_noised = torch.tensor(ca_noised, dtype=cast)
        
        #denote nodes feature as real of fake
        #add dimensions for less noise
        
        expand_node_type = torch.ones_like(self.mask_shift.unsqueeze(-1)).repeat((1,)*len(self.mask_shift.shape)+(self.node_type_dim,))
        expand_node_type =  expand_node_type*self.mask_shift.unsqueeze(-1)
        
        if node_feats is None:
            node_type_noised = np.array([self.ohd.forward_marginal(expand_node_type[i].numpy(),t)[0] for i,t in enumerate(t_vec)])
            node_type_noised = torch.tensor(node_type_noised)
            node_type_noised = torch.roll(node_type_noised,self.randroll,dims=1) #roll=0 if self.roll=False
        else:
            node_type_noised = np.array([self.ohd.forward_marginal(node_feats[i].numpy(),t)[0] for i,t in enumerate(t_vec)])
            node_type_noised = torch.tensor(node_type_noised)
        
        
        bb_noised_out = {'CA': ca_noised, 'N_CA': nc_vec_noised, 'C_CA': cc_vec_noised}
        
        dict_out = {}
        dict_out['bb_shifted'] = bb_shifted
        dict_out['bb_firstp'] = bb_firstp
        dict_out['bb_lastp'] = bb_lastp
        dict_out['bb_noised'] = bb_noised_out 
        
        dict_out['t_vec'] = torch.tensor(t_vec,dtype=cast)
        dict_out['score_scales'] = torch.tensor(score_scales,dtype=cast)
        dict_out['real_nodes_mask'] = torch.roll(self.mask_shift,self.randroll,dims=1)
        dict_out['real_nodes_noise'] = node_type_noised
        dict_out['edge_cons'] = edge_fill
        
        
        #return bb_noised_out, torch.tensor(t_vec,dtype=cast), torch.tensor(score_scales,dtype=cast), torch.tensor(edges_noise,dtype=cast)
        return dict_out
    
    def forward_fixed_nodes(self, bb_dict, t_vec=None, useR3=True , cast=torch.float32):
        
        ca = bb_dict['CA']
        nc_vec = bb_dict['N_CA'].reshape((-1,3))
        cc_vec = bb_dict['C_CA'].reshape((-1,3))
        
        if t_vec is None:
            t_vec =  np.random.uniform(size=ca.shape[0])
        score_scales = [self.score_scaling(t, useR3) for t in t_vec]
        
        #sample rotation
        rot_vec = np.array([self.so3d.sample(t, n_samples=ca.shape[1]) for t in t_vec]).reshape((-1,3))
        rotmat = Rotation.from_rotvec(rot_vec).as_matrix()
        
        batch_shape =  bb_dict['N_CA'].shape
        nc_vec_noised = ru.rot_vec_mul(torch.tensor(rotmat,dtype=cast),nc_vec).reshape(batch_shape)
        cc_vec_noised = ru.rot_vec_mul(torch.tensor(rotmat,dtype=cast),cc_vec).reshape(batch_shape)
        
        #get apply transation to CA
        ca_noised = np.array([self.r3d.forward_marginal(ca[i].numpy(),t)[0] for i,t in enumerate(t_vec)])
        ca_noised = torch.tensor(ca_noised, dtype=cast)
        
        bb_noised_out = {'CA': ca_noised, 'N_CA': nc_vec_noised, 'C_CA': cc_vec_noised}
        
        return bb_noised_out, torch.tensor(t_vec,dtype=cast), torch.tensor(score_scales,dtype=cast)
    
    def forward(self, bb_dict, t=None, useR3=True , cast=torch.float32):
        """single t value"""
        ca = bb_dict['CA']
        nc_vec = bb_dict['N_CA'].reshape((-1,3))
        cc_vec = bb_dict['C_CA'].reshape((-1,3))
        
        if t is None:
            t =  np.random.uniform(size=1)[0]
        score_scales = self.score_scaling(t, useR3)
        
        #sample rotation
        rot_vec = np.array(self.so3d.sample(t, n_samples=ca.shape[0])).reshape((-1,3))
        rotmat = Rotation.from_rotvec(rot_vec).as_matrix()
        
        batch_shape =  bb_dict['N_CA'].shape
        nc_vec_noised = ru.rot_vec_mul(torch.tensor(rotmat,dtype=cast),nc_vec).reshape(batch_shape)
        cc_vec_noised = ru.rot_vec_mul(torch.tensor(rotmat,dtype=cast),cc_vec).reshape(batch_shape)
        
        #get apply transation to CA
        ca_noised = np.array(self.r3d.forward_marginal(ca.numpy(),t)[0])
        ca_noised = torch.tensor(ca_noised, dtype=cast)
        
        bb_noised = {    'CA_noised': ca_noised, 
                         'N_CA_noised': nc_vec_noised,
                         'C_CA_noised': cc_vec_noised,
                         't': torch.tensor(t, dtype=cast),
                         'score_scale': torch.tensor(score_scales,dtype=cast)}
        
        return bb_noised
    
    def score(self, rigids_cur, rigids_init, batched_t):
        trans_score = self.r3d.score(rigids_cur.get_trans().cpu(), 
                                    rigids_init.get_trans().cpu(), 
                                    batched_t[:,None,None].repeat((1,rigids_cur.shape[1],3)).cpu())
        """Probably deprecated."""
        
        
        rots_t = rigids_init.get_rots()
        rots_0 = rigids_cur.get_rots()

        rots_0_inv = rots_0.invert()
        quats_0_inv = rots_0_inv.get_quats()
        quats_t = rots_t.get_quats()
        quats_0t = ru.quat_multiply(quats_0_inv, quats_t)
        rotvec_0t = du.quat_to_rotvec(quats_0t)
        rot_score = self.so3d.torch_score(rotvec_0t, batched_t)
        
        return trans_score, rot_score
    
    def sample_ref(self,batch_size,prot_length=65, device='cuda'):
    
        B = batch_size
        L = prot_length
        cast = torch.float32

        batched_t = torch.tensor(np.ones((B,)),dtype=cast,device=device)
        rot_ref = self.so3d.sample_ref(n_samples=B*L)
        trans_ref = self.r3d.sample_ref(n_samples=B*L)

        nc_vec,cc_vec = rotvec_2_ncVec(rot_ref,B,L,cast = torch.float32)

        noised_dict = {'CA':torch.tensor(trans_ref,dtype=cast).reshape(B,L,-1), 
                       'N_CA':nc_vec.type(cast),
                       'C_CA':cc_vec.type(cast)}

        return noised_dict, batched_t
    
    def forward_single_t(self, bb_dict, t, cast=torch.float32):
        
        ca = bb_dict['CA']
        nc_vec = bb_dict['N_CA'].reshape((-1,3))
        cc_vec = bb_dict['C_CA'].reshape((-1,3))
        #sample rotation
        n_samples = nc_vec.shape[0]
        rotvec = self.so3d.sample(t, n_samples=n_samples)
        rotmat = Rotation.from_rotvec(rotvec).as_matrix()
        #apply rotation
        
        batch_shape =  bb_dict['N_CA'].shape
        
        nc_vec_noised = ru.rot_vec_mul(torch.tensor(rotmat,dtype=cast),nc_vec).reshape(batch_shape)
        cc_vec_noised = ru.rot_vec_mul(torch.tensor(rotmat,dtype=cast),cc_vec).reshape(batch_shape)
        
        #r3d requires numpy
        ca_noised , _  = self.r3d.forward_marginal(ca.numpy(), t)
        ca_noised = torch.tensor(ca_noised, dtype=cast)
        
        bb_noised_out = {'CA': ca_noised, 'N_CA': nc_vec_noised, 'C_CA': cc_vec_noised}
        
        return bb_noised_out
    
 
    