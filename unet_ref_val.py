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
from util import *
from model import *
from graph import *


use_cuda = torch.cuda.is_available()
exp_name = 'unet_scale{}_m{}_rep{}_'.format(int(data.scale), int(m), int(block_reps))
print (exp_name)
#exp_name='unet_scale50_m32_rep1_ResidualBlocks'
#save_name = exp_name + '_ins_point'
#data.batch_size = 4

Voxelize= scn.InputLayer(data.dimension, data.full_scale, mode=4)
Devoxelize = scn.OutputLayer(data.dimension)


def change_lr(optimizer,epoch):
	p = epoch//50
	lr = args.lr * (0.1**p)
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


DIR = 'supervoxel_ref/'

if not os.path.exists(DIR):
	os.mkdir(DIR)

bk_name = exp_name + 'bk_1'
ref_name = exp_name + 'ref_1'


backbone_model = SCN()
ref_model=RefNet_Val()
if use_cuda:
	ref_model=ref_model.cuda()
	backbone_model = backbone_model.cuda()

backbone_model = nn.DataParallel(backbone_model)
checkpoint_restore(backbone_model,bk_name,use_cuda)
checkpoint_restore(ref_model,ref_name,use_cuda)
#sem_optimizer = optim.Adam(unet.parameters())
#training_epoch = 0
total_ious = []
total_tp = [0,0]
total_p = 0
total_max_iou = []
for i,batch in enumerate(data.val_data_loader):
	backbone_model.eval()
	ref_model.eval()
	if use_cuda:
		batch['x'][1]=batch['x'][1].cuda()
		batch['coords']=batch['coords'].cuda()
	with torch.no_grad():
		feats = backbone_model(batch['x'])
		vox_feats = Voxelize([batch['x'][0], feats])
		locs = vox_feats.get_spatial_locations()
		metadata = vox_feats.metadata
		vox_feats = vox_feats.features
		supervoxels = []
		print ('Construct supervoxels...')
		for i in range(data.batch_size):
			each_loc = locs[locs[:,-1]==i][:,0:3]
			each_feat = vox_feats[locs[:,-1]==i]
			grps = grouping(each_feat.cpu().numpy(), each_loc.cpu().numpy(), 1.0, 10, threshold)
			grps = torch.from_numpy(grps)
			grps = grps.unsqueeze(-1) == torch.unique(grps)
			supervoxels.append(grps.long().argmax(-1).unsqueeze(-1))
		supervoxels = torch.cat(supervoxels, 0)
		# Assign supervoxel indexs back to points
		supervoxels = scn.SparseConvNetTensor(supervoxels.float(), metadata, locs)
		# Batch_num_points, 
		supervoxels = Devoxelize(supervoxels)

		supervoxels_ = torch.cat([supervoxels.long(), batch['x'][0][:,-1].unsqueeze(-1)],-1)

		# [(Num_supervoxels, feature_dim), (Num_supervoxels1,feature_dims), ...] len: batch_size
		supvox_f, _ = gather_supervox(feats, supervoxels_)

		# [(Num_supervoxels, 3), (Num_supervoxels1,3), ...] len: batch_size
		supvox_coords, _ = gather_supervox(batch['coords'].float(), supervoxels_)

		batch['supvox_f'] = supvox_f
		batch['supvox_coords'] = supvox_coords
		batch['supervoxels'] = supervoxels.long().squeeze(-1)
		print (supervoxels.shape, batch['x'][0].shape)
		print ('Predict')
		results, ious = ref_model(batch)
		print ('iou', ious.mean())
		total_ious.append(ious)
		#print ('Saving...')
		#loc_supvox = torch.cat([batch['coords'].float().cpu(), batch['x'][1].float().cpu(), supervoxels.float()],-1)
		#for i, name in enumerate(batch['names']):
		#	print (name)
		#	np.save(DIR+name+'_ref', results[i].cpu().numpy())
		#	np.save(DIR+name+'_gt', batch['ref_lbl'][i].cpu().numpy())
		#	np.save(DIR+name+'_pc', loc_supvox[batch['x'][0][:,-1]==i].numpy())
		#	np.save(DIR+name+'_sen', batch['sentences'][i])
total_ious = torch.cat(total_ious, 0)
print ('Overall IOU')
print (total_ious.mean())		
