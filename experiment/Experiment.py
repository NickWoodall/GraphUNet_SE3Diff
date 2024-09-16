from gudiff_model import PDBDataSet_GraphCon
from gudiff_model.Graph_UNet import GraphUNet
from data_rigid_diffuser.diffuser import FrameDiffNoise

from se3_transformer.model.fiber import Fiber
import torch
import os
import logging
from datetime import datetime
from collections import defaultdict
import time
import tree
from se3_transformer.model.FAPE_Loss import FAPE_loss, Qs2Rs, normQ
from torch import einsum
import numpy as np
import se3_diffuse.utils as du
import util.framediff_utils as fu
from data_rigid_diffuser import rigid_utils as ru
import copy
import util.pdb_writer 


class Experiment:

    def __init__(self,
                 conf,
                 ckpt_model=None,
                 cur_step=None,
                 cur_epoch=None,
                 name='gu_null',
                 cast_type=torch.float32,
                 ckpt_opt=None, 
                 swap_metadir=False):
        """Initialize experiment.
        Args:
            exp_cfg: Experiment configuration.
        """
#         with open(config_path, 'r') as file:
#             config = yaml.safe_load(file)
#         conf = Struct(config)
        dt_string = datetime.now().strftime("%dD_%mM_%YY_%Hh_%Mm_%Ss")
        dt_string_short = datetime.now().strftime("%dD_%mM_%YY")
        
        logging.basicConfig(filename=f'{name}_{dt_string_short}.log', level=logging.INFO)
        self._log = logging.getLogger(__name__)
        

        self.name=name
        self._conf = conf
        self.use_cuda = conf['cuda']
        if conf['cuda']:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        self.coord_scale = conf['coord_scale']
        self.N_CA_dist = (PDBDataSet_GraphCon.N_CA_dist/self.coord_scale).to(self.device)
        self.C_CA_dist = (PDBDataSet_GraphCon.C_CA_dist/self.coord_scale).to(self.device)
        self.cast_type = cast_type
        self.swap_metadir=swap_metadir
        
        self.num_epoch = conf['num_epoch']
        self.log_freq = conf['log_freq']
        self.ckpt_freq = conf['ckpt_freq']
        self.early_ckpt = conf['early_chkpt']
        self.t_range = [conf['min_t'], conf['max_t']]
        
        
        self.meta_data_path = conf['meta_data_path']
        self.sample_mode = conf['sample_mode']
        self.B = conf['batch_size']
        self.limit = conf['dataset_max']

        cur_step=conf['trained_steps']
        cur_epoch=conf['epoch']
        
        #graph properties
        self.KNN = conf['KNN']
        self.stride = conf['stride']
        
        #gudiff params
        self.channels_start = conf['channels']
        
        
        self._diffuser = FrameDiffNoise()
        self._graphmaker =  PDBDataSet_GraphCon.Make_KNN_MP_Graphs(mp_stride = self.stride, 
                                                           coord_div = self.coord_scale, 
                                                           cast_type = self.cast_type, 
                                                           channels_start = self.channels_start,
                                                           ndf1= conf['nodefeats_1'], 
                                                           ndf0= conf['nodefeats_0'],
                                                           cuda=conf['cuda']) #cuda is bool True, mod at some point
        #single_t dataset, for testing
        # sd = smallPDBDataset(fdn , meta_data_path = '/mnt/h/datasets/bCov_4H/metadata.csv', 
        #                      filter_dict=False, maxlen=1000, input_t=0.05)
        


        
        self._model = GraphUNet(fiber_start = Fiber({0:12, 1:2}),
                                fiber_out = Fiber({1:2}),
                                batch_size = self.B, 
                                num_layers_ca = conf['num_layers_ca'],
                                k = conf['topk'],
                                stride = conf['stride'],
                                max_degree = 3,
                                channels_div =  conf['channels_div'],
                                num_heads = conf['num_heads'],
                                num_layers = conf['num_layers'],
                                edge_feature_dim = conf['edge_feature_dim'],
                                latent_pool_type = conf['latent_pool_type'],
                                t_size = conf['t_size'],
                                zero_lin = conf['zero_lin'],
                                use_tdeg1 = conf['use_tdeg1'],
                                cuda = conf['cuda']).to(self.device) #cuda is bool True, mod at some point

        

        
        num_parameters = sum(p.numel() for p in self._model.parameters())
        self.num_parameters = num_parameters
        self._log.info(f'Number of model parameters {num_parameters}')
#         self._optimizer = EMA(0.980)
#         for name, param in self._model.named_parameters():
#             if param.requires_grad:
#                 self._optimizer.register(name, param.data)

        if ckpt_model is not None:
            ckpt_model = {k.replace('module.', ''):v for k,v in ckpt_model.items()}
            self._model.load_state_dict(ckpt_model, strict=True)
        
        
        self._optimizer = torch.optim.Adam( self._model.parameters(),
                                                       lr=conf['learning rate'],
                                                       weight_decay=conf['weight_decay'])
        if ckpt_opt is not None:
            self._optimizer.load_state_dict(ckpt_opt)
            fu.optimizer_to(self._optimizer, self.device)
        
        

        self.ckpt_dir =  conf['ckpt_dir']
        self.eval_dir = conf['eval_dir']
        eval_name = f'{self.name}_{dt_string_short}'
        if self.ckpt_dir is not None:
            # Set-up checkpoint location
            ckpt_dir = os.path.join(
                 self.ckpt_dir,
                 self.name,
                 dt_string)
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir, exist_ok=True)
            self.ckpt_dir = ckpt_dir
            self._log.info(f'Checkpoints saved to: {ckpt_dir}')
        else:  
            self._log.info('Checkpoint not being saved.')
            
        if self.eval_dir is not None :
            self.eval_dir = os.path.join(
                self.eval_dir,
                eval_name,
                dt_string)
            self.eval_dir = self.eval_dir
            self._log.info(f'Evaluation saved to: {self.eval_dir}')
        else:
            self.eval_dir = os.devnull
            self._log.info(f'Evaluation will not be saved.')
    #         self._aux_data_history = deque(maxlen=100)
    
        if cur_epoch is None:
            self.trained_epochs = 0
        else:
            self.trained_epochs = cur_epoch
            
        if cur_step is None:
            self.trained_steps = 0
        else:
            self.trained_steps = cur_step
            
    @property
    def diffuser(self):
        return self._diffuser

    @property
    def model(self):
        return self._model

    @property
    def conf(self):
        return self._conf
    
    def create_dataset(self, fake_valid=True):
        
        
        self.dataset = PDBDataSet_GraphCon.smallPDBDataset( self._diffuser , meta_data_path = self.meta_data_path, 
                             filter_dict=True, maxlen=self.limit, t_range=self.t_range, swap_metadir=self.swap_metadir)
        
        self.train_sample = PDBDataSet_GraphCon.TrainSampler(self.B, self.dataset, sample_mode=self.conf['sample_mode'])
        
        train_dL = torch.utils.data.DataLoader(self.dataset, sampler=self.train_sample,
                                                     batch_size=self.B, shuffle=False, collate_fn=None)
        
        if fake_valid:
            valid_dL = train_dL
        else:
            valid_dL = train_dL
            #not implemented yet
        
        return train_dL, valid_dL

    def start_training(self, return_logs=False):


        self._model = self._model.to(self.device)
        self._log.info(f"Using device: {self.device}")
        print(f"Using device: {self.device}")

        self._model.train()
        (train_loader, valid_loader) = self.create_dataset()

        logs = []
        self._log.info(f'number of epochs {self.num_epoch}')
        print('number of epochs', self.num_epoch)
        for epoch in range(self.trained_epochs, self.num_epoch+self.trained_epochs):
            print('epoch', epoch)
            self._log.info(f'epoch {epoch}: start')
            if self.device == 'cuda':
                print('mem_used',torch.cuda.memory_allocated('cuda:0'))
                mem_alloc = torch.cuda.memory_allocated('cuda:0')
                self._log.info(f'mem_used {mem_alloc}')
            #this currently returns nothing
            epoch_log = self.train_epoch(train_loader, valid_loader, epoch=epoch, return_logs=return_logs)
            if return_logs: 
                logs.append(epoch_log)

                
        self.conf['trained_steps']=self.trained_steps
        self.conf['epoch']=self.num_epoch+self.trained_epochs
        ckpt_path = os.path.join(self.ckpt_dir, f'step_{self.trained_steps}.pth')
        du.write_checkpoint(
            ckpt_path,
            copy.deepcopy(self.model.state_dict()),
            self._conf,
            copy.deepcopy(self._optimizer.state_dict()),
            self.trained_epochs,
            self.trained_steps,
            logger=self._log,
            use_torch=True)

        self._log.info('Done')
        return logs

    def train_epoch(self, train_loader, valid_loader,epoch=0, return_logs=False):
        
        log_lossses = defaultdict(list)
    
        global_logs = []
        log_time = time.time()
        step_time = time.time()
        losskeeper = []
        for train_feats in train_loader:
            
            loss, aux_data = self.update_fn(train_feats)
            log_lossses['loss'].append(loss.to('cpu').numpy())
            losskeeper.append(loss.to('cpu').numpy())
            
            self.trained_steps += 1
            # Logging to terminal
            if self.trained_steps == 1 or self.trained_steps % self.log_freq == 0:
                elapsed_time = time.time() - log_time
                log_time = time.time()
                step_per_sec = self.log_freq / elapsed_time
                rolling_losses = tree.map_structure(np.mean, log_lossses)
                loss_log = ' '.join([
                    f'{k}={v[0]:.4f}'
                    for k,v in rolling_losses.items() if 'batch' not in k
                ])
                
                self._log.info(
                    f'[{self.trained_steps}]: {loss_log}, steps/sec={step_per_sec:.5f}')
                log_lossses = defaultdict(list)
                self._log.info(f'{loss} {np.mean(losskeeper[-1000:])}')
                print(f'[{self.trained_steps}]: {loss_log}, steps/sec={step_per_sec:.5f}')
                print(np.mean(losskeeper[-1000:]))

            # Take checkpoint
            
            if self.ckpt_dir is not None and (
                    (self.trained_steps % self.ckpt_freq) == 0
                    or (self.early_ckpt and self.trained_steps == 2)
                ):
                ckpt_path = os.path.join(
                    self.ckpt_dir, f'step_{self.trained_steps}.pth')
                du.write_checkpoint(
                    ckpt_path,
                    copy.deepcopy(self.model.state_dict()),
                    self._conf,
                    copy.deepcopy(self._optimizer.state_dict()),
                    epoch,
                    self.trained_steps,
                    logger=self._log,
                    use_torch=True
                )
                

                # Run evaluation
                self._log.info(f'Running evaluation of {ckpt_path}')
                start_time = time.time()
                eval_dir = os.path.join(self.eval_dir, f'step_{self.trained_steps}')
                print('eval',eval_dir)
                os.makedirs(eval_dir, exist_ok=True)
                ckpt_metrics = self.eval_fn(valid_loader,eval_dir,epoch=epoch)
                eval_time = time.time() - start_time
                self._log.info(f'Finished evaluation in {eval_time:.2f}s')
            else:
                ckpt_metrics = None
                eval_time = None


            if torch.isnan(loss):                
                raise Exception(f'NaN encountered')
                
    def update_fn(self, data):
        """Updates the state using some data and returns metrics."""
        self._optimizer.zero_grad()
        
        batch_feats= tree.map_structure(
                        lambda x: x.to(self.device), data)
        noised_dict =   {'CA': batch_feats['CA_noised'] ,
                         'N_CA': batch_feats['N_CA_noised'].unsqueeze(-2) ,
                         'C_CA': batch_feats['C_CA_noised'].unsqueeze(-2)  }
        
        
        loss, aux_data = self.loss_fn(batch_feats, noised_dict)
        loss.backward()
        self._optimizer.step()
        loss_out = loss.detach().cpu()
        return loss_out , aux_data
    
    def loss_fn(self, batch_feats, noised_dict, t_val=None):
        
        L = batch_feats['CA'].shape[1]
        B = batch_feats['CA'].shape[0]
        CA_t  = batch_feats['CA']
        NC_t = CA_t +  batch_feats['N_CA']
        CC_t = CA_t +  batch_feats['C_CA']
        true =  torch.cat((NC_t,CA_t,CC_t),dim=2).reshape(B,L,3,3)

        CA_n  = batch_feats['CA_noised'].reshape(B, L, 3)
        NC_n = CA_n + batch_feats['N_CA_noised'].reshape(B, L, 3)
        CC_n = CA_n + batch_feats['C_CA_noised'].reshape(B, L, 3)
        noise_xyz =  torch.cat((NC_n,CA_n,CC_n),dim=2).reshape(B,L,3,3)

        x = self._graphmaker.prep_for_network(noised_dict, cuda= self.use_cuda)
        out = self._model(x, batch_feats['t'])
        CA_p = out['1'][:,0,:].reshape(B, L, 3) + CA_n #translation of Calpha
        Qs = out['1'][:,1,:] # rotation of frame
        Qs = Qs.unsqueeze(1).repeat((1,2,1))
        Qs = torch.cat((torch.ones((B*L,2,1),device=Qs.device),Qs),dim=-1).reshape(B,L,2,4)
        Qs = normQ(Qs)
        Rs = Qs2Rs(Qs)
        N_C_to_Rot = torch.cat((noised_dict['N_CA'].reshape(B, L, 3),
                                noised_dict['C_CA'].reshape(B, L, 3)),dim=2).reshape(B,L,2,1,3)

        rot_vecs = einsum('bnkij,bnkhj->bnki',Rs, N_C_to_Rot)
        NC_p = CA_p + rot_vecs[:,:,0,:]*self.N_CA_dist 
        CC_p = CA_p + rot_vecs[:,:,1,:]*self.C_CA_dist 

        pred = torch.cat((NC_p,CA_p,CC_p),dim=2).reshape(B,L,3,3)

        tloss, loss = FAPE_loss(pred.unsqueeze(0), true, batch_feats['score_scale'])

        return tloss, loss #final_loss, aux_loss
    
    
    def generate_tbatch(self, index_in, input_t):
        batch_list = []
        if len(index_in)<1:
            index_in = np.zeros((len(input_t),),dtype=int)
        for i,t in enumerate(input_t):
            if i >= len(index_in):
                batch_list.append(self.dataset.get_specific_t(index_in[-1], input_t[i]))
            else:
                batch_list.append(self.dataset.get_specific_t(index_in[i], input_t[i]))

        batch_feats = {}
        for k in batch_list[0].keys():
            batch_feats[k] = torch.stack([batch_list[i][k] for i in range(len(batch_list))])
            
        return batch_feats
    
    def eval_model(self, batch_feats, noised_dict, t_val=None):
    
        L = batch_feats['CA'].shape[1]
        B = batch_feats['CA'].shape[0]
        CA_t  = batch_feats['CA']
        NC_t = CA_t +  batch_feats['N_CA']
        CC_t = CA_t +  batch_feats['C_CA']
        true =  torch.cat((NC_t,CA_t,CC_t),dim=2).reshape(B,L,3,3)

        CA_n  = batch_feats['CA_noised'].reshape(B, L, 3)
        NC_n = CA_n + batch_feats['N_CA_noised'].reshape(B, L, 3)
        CC_n = CA_n + batch_feats['C_CA_noised'].reshape(B, L, 3)
        noise_xyz =  torch.cat((NC_n,CA_n,CC_n),dim=2).reshape(B,L,3,3)

        x = self._graphmaker.prep_for_network(noised_dict, cuda=self.use_cuda)
        
        with torch.no_grad():
            out = self._model(x, batch_feats['t'])
            CA_p = out['1'][:,0,:].reshape(B, L, 3) + CA_n #translation of Calpha
            Qs = out['1'][:,1,:] # rotation of frame
            Qs = Qs.unsqueeze(1).repeat((1,2,1))
            Qs = torch.cat((torch.ones((B*L,2,1),device=Qs.device),Qs),dim=-1).reshape(B,L,2,4)
            Qs = normQ(Qs)
            Rs = Qs2Rs(Qs)
            N_C_to_Rot = torch.cat((noised_dict['N_CA'].reshape(B, L, 3),
                                    noised_dict['C_CA'].reshape(B, L, 3)),dim=2).reshape(B,L,2,1,3)

            rot_vecs = einsum('bnkij,bnkhj->bnki',Rs, N_C_to_Rot)
            NC_p = CA_p + rot_vecs[:,:,0,:]*self.N_CA_dist 
            CC_p = CA_p + rot_vecs[:,:,1,:]*self.C_CA_dist 

            pred = torch.cat((NC_p,CA_p,CC_p),dim=2).reshape(B,L,3,3)

            tloss, loss = FAPE_loss(pred.unsqueeze(0), true, batch_feats['score_scale'])
            
        NC_t_out = CA_t +  batch_feats['N_CA']*self.N_CA_dist 
        CC_t_out = CA_t +  batch_feats['C_CA']*self.C_CA_dist
        true_out =  torch.cat((NC_t_out,CA_t,CC_t_out),dim=2).reshape(B,L,3,3)
        
        NC_nout = (CA_n + batch_feats['N_CA_noised'].reshape(B, L, 3))*self.N_CA_dist 
        CC_nout = (CA_n + batch_feats['C_CA_noised'].reshape(B, L, 3))*self.C_CA_dist
        noise_out =  torch.cat((NC_nout,CA_n,CC_nout),dim=2).reshape(B,L,3,3)
            
        eval_dict = {'true'  : true_out.to('cpu').numpy()*self.coord_scale,
                    'noise'      : noise_out.to('cpu').numpy()*self.coord_scale,
                    'pred'       : pred.to('cpu').numpy()*self.coord_scale,
                    'loss'       : tloss.to('cpu').numpy()}
            
        return eval_dict
    
    def eval_fn(self, valid_loader, eval_dir, epoch=0, input_t=None, max_cycles=10, protein_length=128):
        
        train_feats = next(iter(valid_loader))

        if input_t is None:
            #visualize_T at selected t-values
            vis_t = np.array([0.01,0.05,0.1,0.2,0.3,0.5,0.8,1.0])
            vis_t = vis_t[None,...].repeat(int(np.ceil(self.B/len(vis_t))),axis=0).flatten()[:self.B]
        elif type(input_t) == float:
            vis_t = np.ones((self.B,))*input_t
        else:
            vis_t = input_t

        index_in = self.train_sample.generate_batch_indices(protein_length=protein_length)
        batch_feats = self.generate_tbatch( index_in,vis_t)

        batch_feats= tree.map_structure(
                        lambda x: x.to(self.device), batch_feats)
        noised_dict =   {'CA': batch_feats['CA_noised'] ,
                         'N_CA': batch_feats['N_CA_noised'].unsqueeze(-2) ,
                         'C_CA': batch_feats['C_CA_noised'].unsqueeze(-2)  }


        eval_dict = self.eval_model(batch_feats,noised_dict)

        util.pdb_writer.dump_tnp_epoch_direc(eval_dict['true'], 
                                            eval_dict['noise'], 
                                            eval_dict['pred'], vis_t, e=epoch, 
                                            numOut=len(vis_t), outdir=eval_dir)
        
        generated = self.reverse(batch_feats['CA'].shape[1], write_out=True)
        losskeeper = []
        eval_steps = 0


        for i,train_feats in enumerate(valid_loader):
            
            batch_feats= tree.map_structure(
                lambda x: x.to(self.device),train_feats)
            noised_dict =   {'CA': batch_feats['CA_noised'] ,
                             'N_CA': batch_feats['N_CA_noised'].unsqueeze(-2) ,
                             'C_CA': batch_feats['C_CA_noised'].unsqueeze(-2)  }

            eval_dict = self.eval_model(batch_feats, noised_dict)
            eval_steps += 1
            losskeeper.append(eval_dict['loss'])   

            if i>max_cycles:
                break
        print('eval_loss',np.mean(losskeeper[-1000:]),len(losskeeper))
        self._log.info(f'eval_loss {np.mean(losskeeper[-1000:])}')
        
        
    def reverse_step(self, noised_dict, batched_t):
        
        
        L = noised_dict['CA'].shape[1]
        B = noised_dict['CA'].shape[0]
    
        CA_n = noised_dict['CA'].to(self.device)

        x = self._graphmaker.prep_for_network(noised_dict,  self.use_cuda)
        
        with torch.no_grad():
            out = self._model(x, batched_t)
            CA_p = out['1'][:,0,:].reshape(B, L, 3) + CA_n #translation of Calpha
            Qs = out['1'][:,1,:] # rotation of frame
            Qs = Qs.unsqueeze(1).repeat((1,2,1))
            Qs = torch.cat((torch.ones((B*L,2,1),device=Qs.device),Qs),dim=-1).reshape(B,L,2,4)
            Qs = normQ(Qs)
            Rs = Qs2Rs(Qs)
            N_C_to_Rot = torch.cat((noised_dict['N_CA'].reshape(B, L, 3),
                                    noised_dict['C_CA'].reshape(B, L, 3)),dim=2).reshape(B,L,2,1,3).to(self.device)

            rot_vecs = einsum('bnkij,bnkhj->bnki',Rs, N_C_to_Rot)
            NC_p = CA_p + rot_vecs[:,:,0,:]*self.N_CA_dist 
            CC_p = CA_p + rot_vecs[:,:,1,:]*self.C_CA_dist 

        return NC_p, CA_p, CC_p
    
    def model_to_device(self):
        self._model = self._model.to(self.device)
        self._log.info(f"Using device: {self.device}")
        print(f"Using device: {self.device}")

    def reverse(self, protein_length, write_out=True):

        L = protein_length
        B = self.B

        noised_dict, batched_t = self._diffuser.sample_ref(self.B, prot_length=L, device=self.device) #implied t=1.0
        pred_dict = {}

        start1 = 1.0
        end1 = 0.3
        dt1 = 0.02

        start2 = 0.3
        end2 = 0.0
        dt2 = 0.0025

        t_list_start = np.arange(start1,end1,-dt1)
        t_list_end = np.arange(start2,end2,-dt2).repeat(2)

        t_list = np.concatenate([t_list_start,t_list_end])

        
        for t in t_list:
            t_vec = np.ones(self.B,)*t
            batched_t = torch.tensor(t_vec,dtype=torch.float32).to(self.device)
            NC_p, CA_p, CC_p = self.reverse_step(noised_dict, batched_t)

            pred = torch.cat((NC_p,CA_p,CC_p),dim=2).reshape(B,L,3,3)
            
            #reset noise dict
            pred_dict['CA'] =   CA_p.cpu().reshape(B, L, 3)
            pred_dict['N_CA'] = PDBDataSet_GraphCon.torch_normalize(NC_p.cpu()-CA_p.cpu()).reshape(B,L,1, 3)
            pred_dict['C_CA'] = PDBDataSet_GraphCon.torch_normalize(CC_p.cpu()-CA_p.cpu()).reshape(B,L,1, 3)
            t_vec = np.ones(self.B,)*t
            batched_t = torch.tensor(t_vec,dtype=torch.float32)
            noised_dict = self._diffuser.forward_single_t(pred_dict, t)


        if write_out:
            predout = pred.to('cpu').numpy()*self.coord_scale
            outdir = os.path.join(self.eval_dir, 'generated/')
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            util.pdb_writer.write_coord_pdb(predout, name=f'gen_L{protein_length}_e{self.num_epoch}',direc=outdir, limit=self.B)
                
        return pred