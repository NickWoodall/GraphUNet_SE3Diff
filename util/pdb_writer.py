import os
import torch
import numpy as np

from gudiff_model.PDBDataSet_GraphCon import  dump_coord_pdb




def write_coord_pdb(xyz_in, name='coords', direc='output/', limit=2):
    
    for i,coords in enumerate(xyz_in):
        if i>limit-1:
            break
        dump_coord_pdb(coords, fileOut=f'{direc}/{name}_{i}.pdb')

def dump_tnp(true, noise, pred, t_val, e=0, numOut=1,outdir='output/'):
    
    if numOut>true.shape[0]:
        numOut = true.shape[0]
    
    for x in range(numOut):
        dump_coord_pdb(true[x], fileOut=f'{outdir}/true_{t_val[x]*100:.0f}_e{e}_{x}.pdb')
        dump_coord_pdb(noise[x], fileOut=f'{outdir}/noise_{t_val[x]*100:.0f}_e{e}_{x}.pdb')
        dump_coord_pdb(pred[x], fileOut=f'{outdir}/pred_{t_val[x]*100:.0f}_e{e}_{x}.pdb')


def dump_tnp_epoch_direc(true, noise, pred, t_val, e:int=0, numOut=1,outdir='output/'):
    
    assert type(e) == int
    if numOut>true.shape[0]:
        numOut = true.shape[0]

    edir = os.path.join(outdir,f'_e{e}')
    if not os.path.exists(edir):
        os.makedirs(os.path.join(edir))
    
    for x in range(numOut):
        dump_coord_pdb(true[x], fileOut=f'{edir}/true_{t_val[x]*100:.0f}_epoch{e}_s{x}.pdb')
        dump_coord_pdb(noise[x], fileOut=f'{edir}/noise_{t_val[x]*100:.0f}_epoch{e}_s{x}.pdb')
        dump_coord_pdb(pred[x], fileOut=f'{edir}/pred_{t_val[x]*100:.0f}_epoch{e}_s{x}.pdb')
        
#deprecated, from failed idea for null/real node types




# def roll2_continous_true(real_mask_in):
#     """Return roll amount to set zero on Nterminal residue for pdb file view"""

#     roll_con_out = []
#     for i,rmr in enumerate(real_mask_in):
#         ep_bool = (rmr^rmr.roll(-1) | rmr^rmr.roll(1)) & rmr
#         si = torch.arange(ep_bool.shape[0])[ep_bool]
#         #circular if start/end real nodes and we need to roll
#         if len(si)<1:
#             roll_con = 0
#         elif rmr[0] and rmr[-1]:
#             #roll last group across barrier
#             roll_con = -si[-1]
#         elif not rmr[0]: #move first group to front
#             roll_con = -si[0]
#         else:
#             roll_con=0

#         roll_con_out.append(roll_con)

#     return roll_con_out


# def pred_to_pdb(pred, pred_mask, pdb_str='pred_revtest',
#                 outdir='output/', coord_scale=10.0, num_out=1):
    
#     pnk_dir = f'{outdir}/reverse_test/'
#     if not os.path.isdir(pnk_dir) and pred_mask is not None:
#         os.makedirs(pnk_dir)
    
#     for x,c in enumerate(pred):
#         if x>= num_out:
#             break
#         p_o = pred[x].detach().to('cpu').numpy()*coord_scale
#         pm = pred_mask[x].detach().to('cpu').numpy()
#         if pm.sum()<1:
#             with open(f'{pnk_dir}/null_{t_val[x]*100:.0f}_e{e}_{x}.txt','w') as f:
#                 f.write(f'These pred masks at index {x} have no members.')
#             continue
#         else:
#             dump_coord_pdb(p_o[pm], fileOut=f'{pnk_dir}/{pdb_str}_{x}.pdb')
    


# def dump_tnp_null(true, noise, pred, t_val,
#                   e=0, 
#                   numOut=1, 
#                   real_mask=None,
#                   pred_mask=None,
#                   outdir='output/', coord_scale=10.0):
    
#     if numOut>true.shape[0]:
#         numOut = true.shape[0]
    
#     tnk_dir = f'{outdir}/true_node_mask/'
#     pnk_dir = f'{outdir}/pred_node_mask/'
#     f_dir = f'{outdir}/full/'
    
#     if not os.path.isdir(tnk_dir) and real_mask is not None:
#         os.makedirs(tnk_dir)
#     if not os.path.isdir(pnk_dir) and pred_mask is not None:
#         os.makedirs(pnk_dir)
#     if not os.path.isdir(f_dir) and real_mask is not None:
#         os.makedirs(f_dir)
    
#     if real_mask is not None:
#         rc = roll2_continous_true(real_mask)
#         for x in range(numOut):
#             t_o = true[x].roll(int(rc[x]),dims=0).detach().to('cpu').numpy()*coord_scale
#             n_o = noise[x].roll(int(rc[x]),dims=0).detach().to('cpu').numpy()*coord_scale
#             p_o = pred[x].roll(int(rc[x]),dims=0).detach().to('cpu').numpy()*coord_scale
#             dump_coord_pdb(t_o, fileOut=f'{f_dir}/true_{t_val[x]*100:.0f}_e{e}_{x}.pdb')
#             dump_coord_pdb(n_o, fileOut=f'{f_dir}/noise_{t_val[x]*100:.0f}_e{e}_{x}.pdb')
#             dump_coord_pdb(p_o, fileOut=f'{f_dir}/pred_{t_val[x]*100:.0f}_e{e}_{x}.pdb')
        
#     if pred_mask is not None:
#         rc = roll2_continous_true(pred_mask)
#         for x,c in enumerate(np.arange(numOut)):
#             t_o = true[x].roll(int(rc[x]),dims=0).detach().to('cpu').numpy()*coord_scale
#             n_o = noise[x].roll(int(rc[x]),dims=0).detach().to('cpu').numpy()*coord_scale
#             p_o = pred[x].roll(int(rc[x]),dims=0).detach().to('cpu').numpy()*coord_scale
#             pm = pred_mask[x].roll(int(rc[x]),dims=0)
#             if pm.sum()<1:
#                 with open(f'{pnk_dir}/null_{t_val[x]*100:.0f}_e{e}_{x}.txt','w') as f:
#                     f.write('These pred masks have no members.')
#                 continue
#             else:
#                 dump_coord_pdb(t_o[pm], fileOut=f'{pnk_dir}/true_{t_val[x]*100:.0f}_e{e}_{x}.pdb')
#                 dump_coord_pdb(n_o[pm], fileOut=f'{pnk_dir}/noise_{t_val[x]*100:.0f}_e{e}_{x}.pdb')
#                 dump_coord_pdb(p_o[pm], fileOut=f'{pnk_dir}/pred_{t_val[x]*100:.0f}_e{e}_{x}.pdb')
            
#     if real_mask is not None:
#         rc = roll2_continous_true(real_mask)
#         for x,c in enumerate(np.arange(numOut)):
#             t_o = true[x].roll(int(rc[x]),dims=0).detach().to('cpu').numpy()*coord_scale
#             n_o = noise[x].roll(int(rc[x]),dims=0).detach().to('cpu').numpy()*coord_scale
#             p_o = pred[x].roll(int(rc[x]),dims=0).detach().to('cpu').numpy()*coord_scale
#             rm = real_mask[x].roll(int(rc[x]),dims=0)
#             dump_coord_pdb(t_o[rm], fileOut=f'{tnk_dir}/true_{t_val[x]*100:.0f}_e{e}_{x}.pdb')
#             dump_coord_pdb(n_o[rm], fileOut=f'{tnk_dir}/noise_{t_val[x]*100:.0f}_e{e}_{x}.pdb')
#             dump_coord_pdb(p_o[rm], fileOut=f'{tnk_dir}/pred_{t_val[x]*100:.0f}_e{e}_{x}.pdb')
            
#     if real_mask is not None:
#         rc = roll2_continous_true(real_mask)
#         for x in range(numOut):
#             dump_coord_pdb(true[x][real_mask[x]], fileOut=f'{tnk_dir}/true_{t_val[x]*100:.0f}_e{e}_{x}.pdb')
#             dump_coord_pdb(noise[x][real_mask[x]], fileOut=f'{tnk_dir}/noise_{t_val[x]*100:.0f}_e{e}_{x}.pdb')
#             dump_coord_pdb(pred[x][real_mask[x]], fileOut=f'{tnk_dir}/pred_{t_val[x]*100:.0f}_e{e}_{x}.pdb')