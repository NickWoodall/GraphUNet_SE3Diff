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
from experiment.Experiment import Experiment
import json
import argparse


with open('configs/base_gun.json','r') as f:
    conf = json.load(f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train Graph U-Net')
    parser.add_argument('name', help='name of run',type=str)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-c", '--config_path',  help='path to config file in json format',type=str)
    group.add_argument("-k", '--checkpoint_file', help='previous checkpoint to load',type=str)

    args = parser.parse_args()

    if args.checkpoint_file is not None:

        ckpt_model=torch.load(args.checkpoint_file)['model']
        ckpt_opt = torch.load(args.checkpoint_file)['optimizer']
        conf = torch.load(args.checkpoint_file)['conf']

        exp = Experiment(conf,name=args.name, ckpt_model=ckpt_model, ckpt_opt=ckpt_opt)

    else:
        with open(args.config_path, 'r') as f:
            conf = json.load(f)
        exp = Experiment(conf, name=args.name, ckpt_model=None, ckpt_opt=None)

    tl, vl = exp.create_dataset() #load dataset from 'meta_data_path.csv'
    exp.start_training()
