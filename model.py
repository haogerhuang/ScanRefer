import torch, torch.nn as nn
import numpy as np
import sparseconvnet as scn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from loss import *
#import data
import data_val as data
from util import *

m = 32
residual_blocks= True
block_reps = 2


class SCN(nn.Module):
	def __init__(self):
		nn.Module.__init__(self)
		self.sparseModel = scn.Sequential().add(
			scn.InputLayer(data.dimension,data.full_scale, mode=4)).add(
			scn.SubmanifoldConvolution(data.dimension, 3, m, 3, False)).add(
			scn.UNet(data.dimension, block_reps, [m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], residual_blocks)).add(
			scn.BatchNormReLU(m)).add(
			#scn.SubmanifoldConvolution(data.dimension, m, 4, 1, False)).add(
			scn.OutputLayer(data.dimension))
		self.linear = nn.Linear(m, m*2)
	def forward(self,x):
		fv=self.sparseModel(x)
		
		y=self.linear(fv)
		#fv = F.normalize(fv, p=2, dim=1)
		return y

class RefNet(nn.Module):
	def __init__(self):
		nn.Module.__init__(self)
		self.gru = nn.GRU(input_size=300, hidden_size=256, batch_first=True)
		#self.feat_fuse = nn.Sequential(nn.Linear(36, 32), nn.ReLU())
		self.feat_fuse = nn.Sequential(nn.Linear(128+3, 128), nn.ReLU())
		#self.lang_sqz = nn.Sequential(nn.Linear(256, 32), nn.ReLU(), nn.Linear(32, 32))
		self.lang_sqz = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
		
		self.lang_cls = nn.Sequential(nn.Linear(64, 18), nn.Dropout())
		self.spvox_cls = nn.Sequential(nn.Linear(64, 20), nn.Dropout())
		#self.heat_map = nn.Sequential(nn.Linear(32, 2), nn.Dropout())
		self.heat_map = nn.Sequential(nn.Linear(128, 2), nn.Dropout())
	def forward(self, batch):
		#fv= self.backbone_model(batch['x'])
		#fv = self.coord_fuse(torch.cat([fv, batch['coords'].float()], 1))	
		total_loss = {}
		total_loss['var'] = 0
		total_loss['dis'] = [0,0]
		total_loss['reg'] = 0
		total_loss['obj'] = 0
		total_loss['lcls'] = 0
		idx = 0
		feat = batch['feat']
		for i, num_p in enumerate(batch['num_points']):
			lang_feat = pack_padded_sequence(batch['lang_feat'][i].float().cuda(), batch['lang_len'][i], batch_first=True, enforce_sorted=False)
			_, lang_feat = self.gru(lang_feat)
			# Num_sentences * 128
			lang_feat = self.lang_sqz(lang_feat.squeeze(0))

			#Language classify
			# Num_sentences * 18
			lang_pred = self.lang_cls(lang_feat)
			total_loss['lcls'] += F.cross_entropy(lang_pred, batch['lang_cls'][i].cuda())

			#lang_feat = self.lang_sqz(lang_feat[-1])
			each_feat = feat[idx:idx+num_p]
			y_ins = batch['y_ins'][idx:idx+num_p]
			y_sem = batch['y'][idx:idx+num_p]
			mask = y_ins > -1
			loss, obj = InsSemLoss(each_feat[mask], y_ins[mask].cuda(), y[mask].cuda(), None, 0.1, 1.5, 1.5)
			# obj_num, d  / _num
			obj_feat, obj_id = obj

			each_coord = batch['coords'][idx:idx+num_p]
			obj_coord, obj_id1 = gather(each_coord[mask], y_ins[mask])

			#each_feat = fvidx:idx+num_p]
			# point -> supervoxel idx  , supervoxel -> instance label idx
			#supvox_ins, supvox_sem = batch['supvox'][1][i]
			# num_supervoxel, 128
			#supvox_feat, insupvox_var = gather_supervox(each_feat, supvox)

			#loss = InsSemLoss(batch['supvox_f'][i], supvox_ins.cuda(), supvox_sem.cuda(), None, 0.1, 1.5, 1.5)

			#supvox_feat = batch['supvox_f'][i]
			#supvox_coords = batch['supvox_coords'][i]
			supvox_feat = torch.cat([supvox_feat, supvox_coords], -1)


			# batch_size * 32	   (batch_size, obj_num) * (obj_num, d)
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
			total_loss['obj'] += torch.nn.functional.cross_entropy(obj_score.view(-1, 2), obj_gt.view(-1), torch.Tensor([0.1, 1]).cuda())
			total_loss['var'] += loss['var']
			total_loss['dis'][0] += loss['dist'][0]
			total_loss['dis'][1] += loss['dist'][1]
			total_loss['reg'] += loss['reg']

		return total_loss

def gather_supervox(feat, supvox, get_var=False):

	# Input: 
	# feat: N, d /  supvox: N, 2
	# Output: results_f: [(N1,d), (N2, d) ... (NB, d)]
	gather_func = scn.InputLayer(1, supvox[:,0].max()+1, mode=4)
	supvox_f = gather_func([supvox, feat])
	supvox_idx = supvox_f.get_spatial_locations()[:,0]
	#supvox_idx, sorted_idx = supvox_idx.sort()
	supvox_batch = supvox_f.get_spatial_locations()[:,1]
	results_f = []
	results_grp = []
	Vars = []
	supvox_f = supvox_f.features
	for i in range(data.batch_size):
		each_supvox_idx = supvox_idx[supvox_batch == i]
		each_supvox_f = supvox_f[supvox_batch==i]
		sort_i = each_supvox_idx.argsort()
		each_supvox_f = each_supvox_f[sort_i]
		each_supvox_idx = each_supvox_idx[sort_i]
		results_f.append(each_supvox_f)
		results_grp.append(each_supvox_idx)
		if get_var:
			mask = supvox[:,1]==i
			each_feat = feat[mask]
			var = torch.norm(each_feat - each_supvox_f[supvox[:,0][mask]], p=2, dim=1)
			Vars.append(var)
	if get_var:
		Vars = torch.cat(Vars, 0)
		Vars = gather_func([supvox, (Vars**2).unsqueeze(-1)])
		Vars = Vars.features.mean()
		
	return results_f, Vars

def gather(feat, lbl):
	gather_func = scn.InputLayer(1, uniq_lbl.shape[0], mode=4)
	grp_f = gather_func([lbl.long().unsqueeze(-1), feat])
	grp_idx = grp_f.get_spatial_locations()[:,0]
	grp_idx, sorted_indice = grp_idx.sort()
	grp_f = grp_f.features[sorted_indice]
	return grp_f, grp_idx


class RefNet_Val(nn.Module):
	def __init__(self):
		nn.Module.__init__(self)
		self.gru = nn.GRU(input_size=300, hidden_size=256, batch_first=True)
		#self.feat_fuse = nn.Sequential(nn.Linear(36, 32), nn.ReLU())
		self.feat_fuse = nn.Sequential(nn.Linear(128+3, 128), nn.ReLU())
		self.lang_sqz = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 64))
		#self.lang_sqz = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
		
		self.lang_cls = nn.Sequential(nn.Linear(128, 18), nn.Dropout())
		self.spvox_cls = nn.Sequential(nn.Linear(64, 20), nn.Dropout())
		#self.heat_map = nn.Sequential(nn.Linear(32, 2), nn.Dropout())
		self.heat_map = nn.Sequential(nn.Linear(128, 2), nn.Dropout())
	def forward(self, batch):
		results = []
		ious = []
		idx = 0
		supervox = batch['supervoxels']
		for i, num_p in enumerate(batch['num_points']):
			lang_feat = pack_padded_sequence(batch['lang_feat'][i].float().cuda(), batch['lang_len'][i], batch_first=True, enforce_sorted=False)
			_, lang_feat = self.gru(lang_feat)
			# Num_sentences * 128
			lang_feat = self.lang_sqz(lang_feat.squeeze(0))

			supvox_feat = batch['supvox_f'][i]
			supvox_coords = batch['supvox_coords'][i]
			supvox_feat = torch.cat([supvox_feat, supvox_coords], -1)
			
			# Mean Shift Clustering 
			grps = MS(supvox_feat, 0.9)
			grps = grps.cuda()

			# Num_supervoxels, Num_sentences, d
			lang_feat_expand = lang_feat.unsqueeze(0).repeat(supvox_feat.shape[0], 1, 1)
			# Num_supervoxels, Num_sentences, d
			supvox_feat_expand = supvox_feat.unsqueeze(1).repeat(1, lang_feat.shape[0], 1)

			overall_feat = self.feat_fuse(torch.cat([supvox_feat_expand, lang_feat_expand], -1))
			# Num_supervoxels, Num_sentences, 2
			obj_score = self.heat_map(overall_feat)
			# Num_supervoxels, Num_sentences
			obj_score = F.softmax(obj_score, -1)[:,:,1]

			# Num_supervoxels, Num_grps
			grps_ = grps.unsqueeze(-1) == torch.unique(grps)
			grps_ = grps_.float()

			# Num_supervoxels, Num_sentences, Num_grps
			sen_grp_score = torch.bmm(obj_score.unsqueeze(-1),grps_.unsqueeze(1))	
			# Num_sentences, Num_grps
			sen_grp_score = sen_grp_score.sum(0)/grps_.sum(0).unsqueeze(0)
			# Num_supervoxels, Num_sentences
			supvox_sen_score = grps_[:,sen_grp_score.argmax(-1)]
			
			#print (obj_score.sum())
			#obj_score = obj_score[supervox[idx:idx+num_p]]

			# Num_points, Num_sentences
			obj_score = supvox_sen_score[supervox[idx:idx+num_p]].long()
			gt = batch['ref_lbl'][i].cuda()
			iou = (obj_score * gt).sum(0).float()/((obj_score | gt).sum(0).float() + 1e-5)
			ious.append(iou.cpu())
			results.append(obj_score)
			idx += num_p
		ious = torch.cat(ious, 0)
		return results, ious


