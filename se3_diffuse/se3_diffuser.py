"""SE(3) diffusion methods."""
import numpy as np
from se3_diffuse import so3_diffuser
from se3_diffuse import r3_diffuser
from scipy.spatial.transform import Rotation
from se3_diffuse import rigid_utils as ru
from se3_diffuse import utils as du
import torch
import logging
import yaml



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




def _extract_trans_rots(rigid: ru.Rigid):
    rot = rigid.get_rots().get_rot_mats().cpu().numpy()
    rot_shape = rot.shape
    num_rots = np.cumprod(rot_shape[:-2])[-1]
    rot = rot.reshape((num_rots, 3, 3))
    rot = Rotation.from_matrix(rot).as_rotvec().reshape(rot_shape[:-2] +(3,))
    tran = rigid.get_trans().cpu().numpy()
    return tran, rot

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

class SE3Diffuser:

    def __init__(self, config_path='se3_diffuse/base.yaml'):
        self._log = logging.getLogger(__name__)

        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self._se3_conf = Struct(config['diffuser'])
        

        self._diffuse_rot = self._se3_conf.diffuse_rot
        self._so3_diffuser = so3_diffuser.SO3Diffuser(self._se3_conf.so3)

        self._diffuse_trans = self._se3_conf.diffuse_trans
        self._r3_diffuser = r3_diffuser.R3Diffuser(self._se3_conf.r3)
        
    def forward_marginal(
            self,
            bb_dict, 
            t_vec=None,
            diffuse_mask: np.ndarray = None,
            as_tensor_7: bool=True,
            cast=torch.float32,
        ):
        """
        Args:
            rigids_0: [..., N] openfold Rigid objects
            t: continuous time in [0, 1].

        Returns:
            rigids_t: [..., N] noised rigid. [..., N, 7] if as_tensor_7 is true. 
            trans_score: [..., N, 3] translation score
            rot_score: [..., N, 3] rotation score
            trans_score_norm: [...] translation score norm
            rot_score_norm: [...] rotation score norm
        """
        
        rigids_0 = ru.Rigid.from_3_points(bb_dict['rigids'][:,:,0,:], 
                        bb_dict['rigids'][:,:,1,:], 
                        bb_dict['rigids'][:,:,2,:])
        
        ca = bb_dict['CA'] #copy of rigids :,:,1,:
        nc_vec = bb_dict['N_CA'].reshape((-1,3)) #norm vector of rigid [:,:,1,:]-[:,:,0,:]
        cc_vec = bb_dict['C_CA'].reshape((-1,3)) #norm vector of rigid [:,:,1,:]-[:,:,2,:]
        
        if t_vec is None:
            t_vec =  np.random.uniform(size=ca.shape[0])
            
        trans_0, rot_0 = _extract_trans_rots(rigids_0)

        rot_score_out = np.zeros_like(rot_0)
        sr_batched = np.zeros_like(rot_0)
        rot_ss = np.zeros_like(t_vec)
        
        trans_t = np.zeros_like(trans_0)
        trans_score_out = np.zeros_like(trans_0)
        trans_ss =  np.zeros_like(t_vec)
        for i,t in enumerate(t_vec):
            trans_t_single, trans_score = self._r3_diffuser.forward_marginal(trans_0[i], t)
            trans_t[i] = trans_t_single
            trans_score_out[i] = trans_score
            trans_ss[i] = self._r3_diffuser.score_scaling(t)
            
            sampled_rots, rot_score = self._so3_diffuser.forward_marginal(rot_0[i], t)
            rot_score_out[i] = rot_score
            sr_batched[i] = sampled_rots
            rot_ss[i] = self._so3_diffuser.score_scaling(t)
        
#         rot_t = du.compose_rotvec(rot_0.reshape((-1,3)), sr_batched.reshape((-1,3))).reshape(rot_0.shape)
#         rigids_t = _assemble_rigid(rot_t, trans_t)
        
        rotmat = Rotation.from_rotvec(sr_batched.reshape(-1,3)).as_matrix()
        batch_shape =  bb_dict['N_CA'].shape# Batch, Length in AA, 3
        nc_vec_noised = ru.rot_vec_mul(torch.tensor(rotmat,dtype=cast),nc_vec).reshape(batch_shape)
        cc_vec_noised = ru.rot_vec_mul(torch.tensor(rotmat,dtype=cast),cc_vec).reshape(batch_shape)
        
        
        

#         if as_tensor_7:
#             rigids_t = rigids_t.to_tensor_7()
        return {
            'trans_score': torch.tensor(trans_score_out,dtype=cast),
            'rot_score': torch.tensor(rot_score_out,dtype=cast),
            'trans_score_scaling': torch.tensor(trans_ss,dtype=cast),
            'rot_score_scaling': torch.tensor(rot_ss,dtype=cast),
            'CA': torch.tensor(trans_t,dtype=cast),
            'N_CA': nc_vec_noised,
            'C_CA': cc_vec_noised,
            'batched_t': torch.tensor(t_vec, dtype=cast)
        }

    def forward_marginal_trans(
            self,
            bb_dict, 
            t_vec=None,
            diffuse_mask: np.ndarray = None,
            as_tensor_7: bool=True,
            cast=torch.float32,
        ):
        """
        Args:
            rigids_0: [..., N] openfold Rigid objects
            t: continuous time in [0, 1].

        Returns:
            rigids_t: [..., N] noised rigid. [..., N, 7] if as_tensor_7 is true. 
            trans_score: [..., N, 3] translation score
            rot_score: [..., N, 3] rotation score
            trans_score_norm: [...] translation score norm
            rot_score_norm: [...] rotation score norm
        """
        
        rigids_0 = ru.Rigid.from_3_points(bb_dict['rigids'][:,:,0,:], 
                        bb_dict['rigids'][:,:,1,:], 
                        bb_dict['rigids'][:,:,2,:])
        
        ca = bb_dict['CA'] #copy of rigids :,:,1,:
        nc_vec = bb_dict['N_CA'].reshape((-1,3)) #norm vector of rigid [:,:,1,:]-[:,:,0,:]
        cc_vec = bb_dict['C_CA'].reshape((-1,3)) #norm vector of rigid [:,:,1,:]-[:,:,2,:]
        
        if t_vec is None:
            t_vec =  np.random.uniform(size=ca.shape[0])
            
        trans_0, rot_0 = _extract_trans_rots(rigids_0)

#         rot_score_out = np.zeros_like(rot_0)
#         sr_batched = np.zeros_like(rot_0)
#         rot_ss = np.zeros_like(t_vec)
        
        trans_t = np.zeros_like(trans_0)
        trans_score_out = np.zeros_like(trans_0)
        trans_ss =  np.zeros_like(t_vec)
        for i,t in enumerate(t_vec):
            trans_t_single, trans_score = self._r3_diffuser.forward_marginal(trans_0[i], t)
            trans_t[i] = trans_t_single
            trans_score_out[i] = trans_score
            trans_ss[i] = self._r3_diffuser.score_scaling(t)
            
#             sampled_rots, rot_score = self._so3_diffuser.forward_marginal(rot_0[i], t)
#             rot_score_out[i] = rot_score
#             sr_batched[i] = sampled_rots
#             rot_ss[i] = self._so3_diffuser.score_scaling(t)
        
#         rot_t = du.compose_rotvec(rot_0.reshape((-1,3)), sr_batched.reshape((-1,3))).reshape(rot_0.shape)
#         rigids_t = _assemble_rigid(rot_t, trans_t)
        
#         rotmat = Rotation.from_rotvec(sr_batched.reshape(-1,3)).as_matrix()
#         batch_shape =  bb_dict['N_CA'].shape# Batch, Length in AA, 3
#         nc_vec_noised = ru.rot_vec_mul(torch.tensor(rotmat,dtype=cast),nc_vec).reshape(batch_shape)
#         cc_vec_noised = ru.rot_vec_mul(torch.tensor(rotmat,dtype=cast),cc_vec).reshape(batch_shape)
        
        
        trans_score_out = torch.tensor(trans_score_out,dtype=cast)
        trans_tss = torch.tensor(trans_ss,dtype=cast)
        
#         if as_tensor_7:
#             rigids_t = rigids_t.to_tensor_7()
        return {
            'trans_score': trans_score_out,
            'rot_score': torch.zeros_like(trans_score_out ,dtype=cast),
            'trans_score_scaling': torch.tensor(trans_ss,dtype=cast),
            'rot_score_scaling': torch.zeros_like(trans_tss,dtype=cast),
            'CA': torch.tensor(trans_t,dtype=cast),
            'N_CA': bb_dict['N_CA'].type(cast),
            'C_CA': bb_dict['C_CA'].type(cast),
            'batched_t': torch.tensor(t_vec, dtype=cast)
        }

    def forward_marginal_prev(
            self,
            rigids_0: ru.Rigid,
            t: float,
            diffuse_mask: np.ndarray = None,
            as_tensor_7: bool=True,
        ):
        """
        Args:
            rigids_0: [..., N] openfold Rigid objects
            t: continuous time in [0, 1].

        Returns:
            rigids_t: [..., N] noised rigid. [..., N, 7] if as_tensor_7 is true. 
            trans_score: [..., N, 3] translation score
            rot_score: [..., N, 3] rotation score
            trans_score_norm: [...] translation score norm
            rot_score_norm: [...] rotation score norm
        """
        trans_0, rot_0 = _extract_trans_rots(rigids_0)

        if not self._diffuse_rot:
            rot_t, rot_score, rot_score_scaling = (
                rot_0,
                np.zeros_like(rot_0),
                np.ones_like(t)
            )
        else:
            rot_t, rot_score = self._so3_diffuser.forward_marginal(
                rot_0, t)
            rot_score_scaling = self._so3_diffuser.score_scaling(t)

        if not self._diffuse_trans:
            trans_t, trans_score, trans_score_scaling = (
                trans_0,
                np.zeros_like(trans_0),
                np.ones_like(t)
            )
        else:
            trans_t, trans_score = self._r3_diffuser.forward_marginal(
                trans_0, t)
            trans_score_scaling = self._r3_diffuser.score_scaling(t)

        if diffuse_mask is not None:
            # diffuse_mask = torch.tensor(diffuse_mask).to(rot_t.device)
            rot_t = self._apply_mask(
                rot_t, rot_0, diffuse_mask[..., None])
            trans_t = self._apply_mask(
                trans_t, trans_0, diffuse_mask[..., None])

            trans_score = self._apply_mask(
                trans_score,
                np.zeros_like(trans_score),
                diffuse_mask[..., None])
            rot_score = self._apply_mask(
                rot_score,
                np.zeros_like(rot_score),
                diffuse_mask[..., None])
        rigids_t = _assemble_rigid(rot_t, trans_t)
        if as_tensor_7:
            rigids_t = rigids_t.to_tensor_7()
        return {
            'rigids_t': rigids_t,
            'trans_score': trans_score,
            'rot_score': rot_score,
            'trans_score_scaling': trans_score_scaling,
            'rot_score_scaling': rot_score_scaling,
        }

    def calc_trans_0(self, trans_score, trans_t, t):
        return self._r3_diffuser.calc_trans_0(trans_score, trans_t, t)

    def calc_trans_score(self, trans_t, trans_0, t, use_torch=False, scale=True):
        return self._r3_diffuser.score(
            trans_t, trans_0, t, use_torch=use_torch, scale=scale)

    def calc_rot_score(self, rots_t, rots_0, t):
        rots_0_inv = rots_0.invert()
        quats_0_inv = rots_0_inv.get_quats()
        quats_t = rots_t.get_quats()
        quats_0t = ru.quat_multiply(quats_0_inv, quats_t)
        rotvec_0t = du.quat_to_rotvec(quats_0t)
        return self._so3_diffuser.torch_score(rotvec_0t, t)

    def _apply_mask(self, x_diff, x_fixed, diff_mask):
        return diff_mask * x_diff + (1 - diff_mask) * x_fixed

    def trans_parameters(self, trans_t, score_t, t, dt, mask):
        return self._r3_diffuser.distribution(
            trans_t, score_t, t, dt, mask)

    def score(
            self,
            rigid_0: ru.Rigid,
            rigid_t: ru.Rigid,
            t: float):
        tran_0, rot_0 = _extract_trans_rots(rigid_0)
        tran_t, rot_t = _extract_trans_rots(rigid_t)

        if not self._diffuse_rot:
            rot_score = np.zeros_like(rot_0)
        else:
            rot_score = self._so3_diffuser.score(
                rot_t, t)

        if not self._diffuse_trans:
            trans_score = np.zeros_like(tran_0)
        else:
            trans_score = self._r3_diffuser.score(tran_t, tran_0, t)

        return trans_score, rot_score

    def score_scaling(self, t):
        rot_score_scaling = self._so3_diffuser.score_scaling(t)
        trans_score_scaling = self._r3_diffuser.score_scaling(t)
        return rot_score_scaling, trans_score_scaling

    def reverse(
            self,
            rigid_t: ru.Rigid,
            rot_score: np.ndarray,
            trans_score: np.ndarray,
            t: float,
            dt: float,
            diffuse_mask: np.ndarray = None,
            center: bool=True,
            noise_scale: float=1.0,
        ):
        """Reverse sampling function from (t) to (t-1).

        Args:
            rigid_t: [..., N] protein rigid objects at time t.
            rot_score: [..., N, 3] rotation score.
            trans_score: [..., N, 3] translation score.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: [..., N] which residues to update.
            center: true to set center of mass to zero after step

        Returns:
            rigid_t_1: [..., N] protein rigid objects at time t-1.
        """
        trans_t, rot_t = _extract_trans_rots(rigid_t)
        if not self._diffuse_rot:
            rot_t_1 = rot_t
        else:
            rot_t_1 = self._so3_diffuser.reverse(
                rot_t=rot_t,
                score_t=rot_score,
                t=t,
                dt=dt,
                noise_scale=noise_scale,
                )
        if not self._diffuse_trans:
            trans_t_1 = trans_t
        else:
            trans_t_1 = self._r3_diffuser.reverse(
                x_t=trans_t,
                score_t=trans_score,
                t=t,
                dt=dt,
                center=center,
                noise_scale=noise_scale
                )

        if diffuse_mask is not None:
            trans_t_1 = self._apply_mask(
                trans_t_1, trans_t, diffuse_mask[..., None])
            rot_t_1 = self._apply_mask(
                rot_t_1, rot_t, diffuse_mask[..., None])

        return _assemble_rigid(rot_t_1, trans_t_1)

    def sample_ref(
            self,
            n_samples: int,
            impute: ru.Rigid=None,
            diffuse_mask: np.ndarray=None,
            as_tensor_7: bool=False
        ):
        """Samples rigids from reference distribution.

        Args:
            n_samples: Number of samples.
            impute: Rigid objects to use as imputation values if either
                translations or rotations are not diffused.
        """
        if impute is not None:
            assert impute.shape[0] == n_samples
            trans_impute, rot_impute = _extract_trans_rots(impute)
            trans_impute = trans_impute.reshape((n_samples, 3))
            rot_impute = rot_impute.reshape((n_samples, 3))
            trans_impute = self._r3_diffuser._scale(trans_impute)

        if diffuse_mask is not None and impute is None:
            raise ValueError('Must provide imputation values.')

        if (not self._diffuse_rot) and impute is None:
            raise ValueError('Must provide imputation values.')

        if (not self._diffuse_trans) and impute is None:
            raise ValueError('Must provide imputation values.')

        if self._diffuse_rot:
            rot_ref = self._so3_diffuser.sample_ref(
                n_samples=n_samples)
        else:
            rot_ref = rot_impute

        if self._diffuse_trans:
            trans_ref = self._r3_diffuser.sample_ref(
                n_samples=n_samples
            )
        else:
            trans_ref = trans_impute

        if diffuse_mask is not None:
            rot_ref = self._apply_mask(
                rot_ref, rot_impute, diffuse_mask[..., None])
            trans_ref = self._apply_mask(
                trans_ref, trans_impute, diffuse_mask[..., None])
        trans_ref = self._r3_diffuser._unscale(trans_ref)
        rigids_t = _assemble_rigid(rot_ref, trans_ref)
        if as_tensor_7:
            rigids_t = rigids_t.to_tensor_7()
        return {'rigids_t': rigids_t}
