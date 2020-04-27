# Copyright 2016-present, Facebook, Inc.
 #All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Options
m = 32 # 16 or 32
residual_blocks=True #True or False
block_reps = 2 #Conv block repetition factor: 1 or 2
num_classes = 20

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sparseconvnet as scn
import time
import os, sys, glob
import math
import argparse
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
import itertools
from loss import *
from model import *
from util import *
use_cuda = torch.cuda.is_available()
bk_name='unet_scale50_m32_rep2_ins_0.1_1.5_1.5'
exp_name = 'unet_scale{}_m{}_rep{}_64d_'.format(int(data.scale), int(m), int(block_reps))
exp_name = 'unet_scale{}_m{}_rep{}_'.format(int(data.scale), int(m), int(block_reps))

EXT = 'WORotate_'
print (exp_name + EXT)
#exp_name='unet_scale50_m32_rep1_ResidualBlocks'
#save_name = exp_name + '_ins_point'
#data.batch_size = 4

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3)
args= parser.parse_args()

	
def change_lr(optimizer,epoch):
	p = epoch//50
	lr = args.lr * (0.1**p)
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

backbone_model = SCN.cuda()
backbone_model = nn.DataParallel(backbone_model)
ref_model=RefNet()
if use_cuda:
	ref_model=ref_model.cuda()

#model = nn.DataParallel(model)
training_epochs=512
#training_epoch=scn.checkpoint_restore(model,exp_name,'unet',use_cuda)
#training_epoch = checkpoint_restore(backbone_model,exp_name+'sem',use_cuda)
#training_epoch = checkpoint_restore(ref_model,exp_name+'ref_1',use_cuda)
training_epoch = 0

backbone_model.module.linear = nn.Linear(32, 64).cuda()
#print (backbone_model.module)
#model.backbone_model = backbone_model
optimizer = optim.Adam(list(backbone_model.parameters())+list(ref_model.parameters()),lr=args.lr)
#sem_optimizer = optim.Adam(unet.parameters())
#training_epoch = 0
for epoch in range(training_epoch, training_epochs+1):
	backbone_model.train()
	ref_model.train()
	change_lr(optimizer, epoch)
	start = time.time()
	train_loss=0
	total_sem_loss = 0
	total_var_loss = 0
	total_dis_loss = 0
	total_pos_loss = 0
	total_neg_loss = 0
	total_reg_loss = 0
	total_obj_loss = 0
	total_supvox_loss = 0
	iteration = 0
	for i,batch in enumerate(data.train_data_loader):
		optimizer.zero_grad()
		if use_cuda:
			batch['x'][1]=batch['x'][1].cuda()
			batch['coords']=batch['coords'].cuda()
		fv = backbone_model(batch['x'])
		supvox_f, supvox_var = gather_supervox(fv, batch['supvox'][0], True)
		supvox_coords, _ = gather_supervox(batch['coords'].float(), batch['supvox'][0])	
		batch['supvox_f'] = supvox_f
		batch['supvox_coords'] = supvox_coords
		loss = ref_model(batch)
		total_loss = loss['var']+loss['dis'][0]+loss['dis'][1]+loss['reg']+loss['obj']+supvox_var
		total_loss /= data.batch_size
		#print (loss['var_loss']/data.batch_size, loss['dis_loss']/data.batch_size)
		#loss = torch.nn.functional.cross_entropy(predictions,batch['ref_lbl'])
		#i_loss = 0
		#s_loss, i_loss = block_train(optimizer, unet, batch['x'], batch['y'], batch['y_ins'], batch['coords'], instance_loss, 1.0, 1.0)
		#if batch['y'].size(0) > 1000000:
		#	 indices = np.random.choice(batch['y'].size(0), 1000000, replace=False)
		#else:
		#	 indices = np.arange(batch['y'].size(0))
		#loss = sem_loss + i_loss/data.batch_size
		#loss = sem_loss
		total_var_loss += loss['var'].item()/data.batch_size
		total_pos_loss += loss['dis'][0].item()/data.batch_size
		total_neg_loss += loss['dis'][1].item()/data.batch_size
		total_reg_loss += loss['reg'].item()/data.batch_size
		total_obj_loss += loss['obj'].item()/data.batch_size
		total_supvox_loss += supvox_var.item()
		#total_ins_loss += i_loss.item()/data.batch_size
		#print (loss)
		#print (i_loss)
		total_loss.backward()
		optimizer.step()
	print(epoch,'Train loss {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} s'.format(total_var_loss/i, total_pos_loss/i, total_neg_loss/i, total_reg_loss/i,total_obj_loss/i, total_supvox_loss/i, time.time() - start))
	checkpoint_save(backbone_model,exp_name+EXT+'bk',epoch, use_cuda)
	checkpoint_save(ref_model,exp_name+EXT+'ref',epoch, use_cuda)
	
