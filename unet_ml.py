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

import torch, data
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
use_cuda = torch.cuda.is_available()
bk_name='unet_scale50_m32_rep2_ins_0.1_1.5_1.5'
exp_name = 'unet_scale{}_m{}_rep{}_128d_'.format(int(data.scale), int(m), int(block_reps))
print (exp_name)
#exp_name='unet_scale50_m32_rep1_ResidualBlocks'
#save_name = exp_name + '_ins_point'
#data.batch_size = 4

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3)
args= parser.parse_args()
class SCN1(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.sparseModel = scn.Sequential().add(
           scn.InputLayer(data.dimension,data.full_scale, mode=4)).add(
           scn.SubmanifoldConvolution(data.dimension, 3, m, 3, False)).add(
               scn.UNet(data.dimension, block_reps, [m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], residual_blocks)).add(
           scn.BatchNormReLU(m)).add(
           scn.OutputLayer(data.dimension))
        self.linear = nn.Linear(m, num_classes)
    def forward(self,x):
        fv=self.sparseModel(x)
        
        y=self.linear(fv)
        #fv = F.normalize(fv, p=2, dim=1)
        return fv, y

class SCN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.sparseModel = scn.Sequential().add(
           scn.InputLayer(data.dimension,data.full_scale, mode=4)).add(
           scn.SubmanifoldConvolution(data.dimension, 3, m, 3, False)).add(
               scn.UNet(data.dimension, block_reps, [m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], residual_blocks)).add(
           scn.BatchNormReLU(m)).add(
           scn.SubmanifoldConvolution(data.dimension, m, 4*m, 1, False)).add(
           scn.OutputLayer(data.dimension))
    def forward(self,x):
        fv=self.sparseModel(x)
        
        #y=self.linear(fv)
        #fv = F.normalize(fv, p=2, dim=1)
        return fv

class RefNet(nn.Module):
	def __init__(self):
		nn.Module.__init__(self)
		self.backbone_model = None
		#self.model = self.model.cuda()
		#self.model = nn.DataParallel(self.model)
		#self.coord_fuse = nn.Sequential(nn.Linear(64+3, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
		self.gru = nn.GRU(input_size=300, hidden_size=256, batch_first=True)
		#self.feat_fuse = nn.Sequential(nn.Linear(36, 32), nn.ReLU())
		self.feat_fuse = nn.Sequential(nn.Linear(256, 128), nn.ReLU())
		#self.lang_sqz = nn.Sequential(nn.Linear(256, 32), nn.ReLU(), nn.Linear(32, 32))
		self.lang_sqz = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
		
		self.lang_cls = nn.Sequential(nn.Linear(64, 20), nn.Dropout())
		self.heat_map = nn.Sequential(nn.Linear(128, 2), nn.Dropout())
		self.gather_func = scn.InputLayer
	def forward(self, batch, fv):
		#fv= self.backbone_model(batch['x'])
		#fv = self.coord_fuse(torch.cat([fv, batch['coords'].float()], 1))	
		var_loss = 0
		dis_loss = [0,0]
		reg_loss = 0
		obj_loss = 0
		idx = 0
		for i, num_p in enumerate(batch['num_points']):
			lang_feat = pack_padded_sequence(batch['lang_feat'][i].float().cuda(), batch['lang_len'][i], batch_first=True, enforce_sorted=False)
			_, lang_feat = self.gru(lang_feat)
			# Num_sentences * 128
			lang_feat = self.lang_sqz(lang_feat.squeeze(0))
			loss = InsSemLoss(fv[idx:idx+num_p], batch['y_ins'][idx:idx+num_p].cuda(), batch['y'][idx:idx+num_p].cuda(), None, 0.1, 1.5, 1.5)
			# obj_num, d  / obj_num
			#obj_feat, obj_id = data['obj_feat']

			# point -> supervoxel idx  , supervoxel -> instance label idx
			supvox, supvox_ins = batch['supvox'][i]
			# num_supervoxel, 128
			supvox_feat = self.gather_supervox(fv[idx:idx+num_p], supvox, supvox_ins)

			# batch_size * 32      (batch_size, obj_num) * (obj_num, d)
			#obj_feat = torch.matmul((batch['lang_objID'][i].cuda().unsqueeze(-1)==obj_id).float(), obj_feat)
			# Num_sentences, obj_num, d
			#lang_feat_expand = lang_feat.unsqueeze(1).repeat(1, supvox_feat.shape[0], 1)
			# Num_sentences, obj_num, d
			#supvox_feat_expand = supvox_feat.unsqueeze(0).repeat(lang_feat.shape[0], 1, 1)

			# Num_supervoxels, Num_sentences, d
			lang_feat_expand = lang_feat.unsqueeze(0).repeat(supvox_feat.shape[0], 1, 1)
			# Num_supervoxels, Num_sentences, d
			supvox_feat_expand = supvox_feat.unsqueeze(1).repeat(1, lang_feat.shape[0], 1)

			overall_feat = self.feat_fuse(torch.cat([supvox_feat_expand, lang_feat_expand], -1))
			# Num_supervoxels, Num_sentences, 2
			obj_score = self.heat_map(overall_feat)
			# gt (Num_sentences, obj_num)
			#obj_gt = (batch['lang_objID'][i].unsqueeze(-1).cuda() == obj_id).long()

			# Num-supervoxels, Num_sentences
			obj_gt = batch['ref_lbl'][i].cuda()
			obj_loss += torch.nn.functional.cross_entropy(obj_score.view(-1, 2), obj_gt.view(-1), torch.Tensor([0.1, 1]).cuda())
			var_loss += loss['var']
			dis_loss[0] += loss['dist'][0]
			dis_loss[1] += loss['dist'][1]
			reg_loss += loss['reg']
			idx += num_p

		return {'var': var_loss, 'dis': dis_loss, 'reg': reg_loss, 'obj': obj_loss}
	def gather_supervox(self, feat, supvox, supvox_ins):
		gather_func = self.gather_func(1, supvox_ins.shape[0], mode=4)
		supvox_f = gather_func([supvox.unsqueeze(-1), feat])
		supvox_idx = supvox_f.get_spatial_locations()[:,0]
		supvox_f = supvox_f.features
		supvox_f = supvox_f[supvox_idx.argsort()] 
		return supvox_f


def checkpoint_save(model,exp_name,epoch, use_cuda=True):
    f=exp_name+'-%09d'%epoch+'.pth'
    model.cpu()
    torch.save(model.state_dict(),f)
    if use_cuda:
        model.cuda()
    #remove previous checkpoints unless they are a power of 2 to save disk space
    epoch=epoch-1
    f=exp_name+'-%09d'%epoch+'.pth'
    if os.path.isfile(f):
        if not is_power2(epoch):
            os.remove(f)

def checkpoint_restore(model,exp_name,use_cuda=True,epoch=0):
    if use_cuda:
        model.cpu()
    if epoch>0:
        f=exp_name+'-%09d'%epoch+'.pth'
        assert os.path.isfile(f)
        print('Restore from ' + f)
        model.load_state_dict(torch.load(f))
    else:
        f=sorted(glob.glob(exp_name+'-*.pth'))
        if len(f)>0:
            f=f[-1]
            print('Restore from ' + f)
            model.load_state_dict(torch.load(f))
            epoch=int(f[len(exp_name)+1:-4])
    if use_cuda:
        model.cuda()
    return epoch+1

def is_power2(num):
	return num != 0 and ((num & (num - 1)) == 0)
	
def gen_blocks(coords, block_size, stride):
    #coords = coords.numpy()
    Mx, My, _ = coords.max(0)
    mx, my, _ = coords.min(0)
    Mx = np.ceil(Mx) - stride
    My = np.ceil(My) - stride
    mx = (np.floor(mx) + np.rint(mx))/2
    my = (np.floor(my) + np.rint(my))/2
    blocks = [(x,y) for x in np.arange(mx, Mx, stride) for y in np.arange(my, My, stride)]
    return blocks

def change_lr(optimizer,epoch):
	p = epoch//50
	lr = args.lr * (0.1**p)
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

backbone_model = SCN().cuda()
backbone_model = nn.DataParallel(backbone_model)
ref_model=RefNet()
if use_cuda:
    ref_model=ref_model.cuda()

#model = nn.DataParallel(model)
training_epochs=512
#training_epoch=scn.checkpoint_restore(model,exp_name,'unet',use_cuda)
training_epoch = 0
training_epoch = checkpoint_restore(backbone_model,exp_name+'bk',use_cuda)
training_epoch = checkpoint_restore(ref_model,exp_name+'ref',use_cuda)
#model.backbone_model = backbone_model
optimizer = optim.Adam(list(backbone_model.parameters())+list(ref_model.parameters()),lr=args.lr)
#sem_optimizer = optim.Adam(unet.parameters())
#training_epoch = 0
for epoch in range(training_epoch, training_epochs+1):
    backbone_model.train()
    ref_model.train()
    change_lr(optimizer, epoch)
    stats = {}
    scn.forward_pass_multiplyAdd_count=0
    scn.forward_pass_hidden_states=0
    start = time.time()
    train_loss=0
    total_sem_loss = 0
    total_var_loss = 0
    total_dis_loss = 0
    total_pos_loss = 0
    total_neg_loss = 0
    total_reg_loss = 0
    total_obj_loss = 0
    iteration = 0
    for i,batch in enumerate(data.train_data_loader):

        optimizer.zero_grad()
        if use_cuda:
            batch['x'][1]=batch['x'][1].cuda()
            batch['coords']=batch['coords'].cuda()
        fv = backbone_model(batch['x'])
        loss = ref_model(batch, fv)
        total_loss = loss['var'] + loss['dis'][0] + loss['dis'][1] + loss['reg'] + loss['obj']
        total_loss /= data.batch_size
        #print (loss['var_loss']/data.batch_size, loss['dis_loss']/data.batch_size)
        #loss = torch.nn.functional.cross_entropy(predictions,batch['ref_lbl'])
        #i_loss = 0
        #s_loss, i_loss = block_train(optimizer, unet, batch['x'], batch['y'], batch['y_ins'], batch['coords'], instance_loss, 1.0, 1.0)
        #if batch['y'].size(0) > 1000000:
        #    indices = np.random.choice(batch['y'].size(0), 1000000, replace=False)
        #else:
        #    indices = np.arange(batch['y'].size(0))
        #loss = sem_loss + i_loss/data.batch_size
        #loss = sem_loss
        total_var_loss += loss['var'].item()/data.batch_size
        total_pos_loss += loss['dis'][0].item()/data.batch_size
        total_neg_loss += loss['dis'][1].item()/data.batch_size
        total_reg_loss += loss['reg'].item()/data.batch_size
        total_obj_loss += loss['obj'].item()/data.batch_size
        #total_ins_loss += i_loss.item()/data.batch_size
        #print (loss)
        #print (i_loss)
        total_loss.backward()
        optimizer.step()
    print(epoch,'Train loss {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} s'.format(total_var_loss/i, total_pos_loss/i, total_neg_loss/i, total_reg_loss/i,total_obj_loss/i, time.time() - start))
    checkpoint_save(backbone_model,exp_name+'bk',epoch, use_cuda)
    checkpoint_save(ref_model,exp_name+'ref',epoch, use_cuda)
    
