# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Options
scale=50  #Voxel size = 1/scale
val_reps=1 # Number of test views, 1 or more
batch_size=2
lang_batch_size=8
elastic_deformation=False

#Supervoxels
scale1=50
K = 1
min_size = 5

MAX_DES_LEN = 30
GLOVE_PICKLE = 'glove.p'

import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp, time, json, pickle, random
import sparseconvnet as scn
from graph import *

dimension=3
full_scale=4096 #Input field size
center=False

Voxelize = scn.InputLayer(dimension, full_scale, mode=4)
Devoxelize = scn.OutputLayer(dimension)

# Class IDs have been mapped to the range {0,1,...,19}
#NYU_CLASS_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
remap = {}

f = open('labelids.txt', 'r')
NYU_CLASS_IDS = f.readlines()[2:]

for i, line in enumerate(NYU_CLASS_IDS):
    obj_name = line.strip().split('\t')[-1]
    remap[obj_name] = i

train_3d = {}
supervoxel = {}
def load_data(name):
	idx = name[0].find('scene')
	scene_name = name[0][idx:idx+12]
	return torch.load(name[0]), scene_name

for x in torch.utils.data.DataLoader(
        glob.glob('train/*.pth'),
        collate_fn=load_data, num_workers=mp.cpu_count()):
    train_3d[x[1]] = x[0]

for x in torch.utils.data.DataLoader(
        glob.glob('supervoxel50_f2/*.pth'),
        collate_fn=load_data, num_workers=mp.cpu_count()):
    supervoxel[x[1]] = x[0]
print('Training examples:', len(train_3d))

scanrefer = json.load(open('ScanRefer_filtered_train.json'))

with open(GLOVE_PICKLE, 'rb') as f:
	glove = pickle.load(f)
lang = {}

for i, data in enumerate(scanrefer):
	scene_id = data['scene_id']
	object_id = data['object_id']
	ann_id = data['ann_id']

	if scene_id not in lang:
		lang[scene_id] = {'idx':[]}
	if object_id not in lang[scene_id]:
		lang[scene_id][object_id] = {}
	tokens = data['token']
	embeddings = np.zeros((MAX_DES_LEN, 300))
	for token_id in range(MAX_DES_LEN):
		if token_id < len(tokens):
			token = tokens[token_id]
			if token in glove:
				embeddings[token_id] = glove[token]
			else:
				embeddings[token_id] = glove['unk']
		lang[scene_id][object_id][ann_id] = [embeddings, len(tokens)]
	
	lang[scene_id]['idx'].append(i)
	
#loader_list = sorted(list(train_3d.keys()))[0:32]
loader_list = list(train_3d.keys())
#for i in train_3d.keys():
#	multiplier = len(lang[i]['idx'])//lang_batch_size + 1
#	loader_list += [i]*multiplier

#train = train[0:32]
#train_name = train_name[0:32]
#val = val[0:32]
#Elastic distortion
blur0=np.ones((3,1,1)).astype('float32')/3
blur1=np.ones((1,3,1)).astype('float32')/3
blur2=np.ones((1,1,3)).astype('float32')/3
def elastic(x,gran,mag):
    bb=np.abs(x).max(0).astype(np.int32)//gran+3
    noise=[np.random.randn(bb[0],bb[1],bb[2]).astype('float32') for _ in range(3)]
    noise=[scipy.ndimage.filters.convolve(n,blur0,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur1,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur2,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur0,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur1,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur2,mode='constant',cval=0) for n in noise]
    ax=[np.linspace(-(b-1)*gran,(b-1)*gran,b) for b in bb]
    interp=[scipy.interpolate.RegularGridInterpolator(ax,n,bounds_error=0,fill_value=0) for n in noise]
    def g(x_):
        return np.hstack([i(x_)[:,None] for i in interp])
    return x+g(x)*mag


def trainMerge(tbl):
    locs=[]
    feats=[]
    labels=[]
    ins_labels=[]
    ref_labels=[]
    coords=[]
    num_points=[]
    scene_names=[]
    batch_lang_feat=[]
    batch_lang_len=[]
    batch_lang_objID=[]
    batch_lang_cls=[]
    batch_ref_lbl=[]
    batch_supvox=[]
    batch_supvox_lbl=[]
    for idx,scene_id in enumerate(tbl):
        
        #scene_id = scanrefer[i]['scene_id']  
        #object_id = scanrefer[i]['object_id']
        #ann_id = scanrefer[i]['ann_id']
        
        scene_dict = lang[scene_id]
        refer_idxs = lang[scene_id]['idx']
        lang_feat=[]
        lang_len=[]
        lang_objID=[]
        lang_cls=[]
        #if len(refer_idxs) > 20:
        #    refer_idxs = np.random.choice(refer_idxs, 20, replace=False)
        for i in refer_idxs:
            scene_id = scanrefer[i]['scene_id']  
            object_id = scanrefer[i]['object_id']
            ann_id = scanrefer[i]['ann_id']
            object_name = scanrefer[i]['object_name']
            lang_feat.append(torch.from_numpy(lang[scene_id][object_id][ann_id][0])) 
            lang_len.append(min(MAX_DES_LEN, lang[scene_id][object_id][ann_id][1]))
            lang_objID.append(int(object_id))
            if object_name not in remap:
                lang_cls.append(-100)
            else:
                lang_cls.append(remap[object_name])
        # Obj_num, 30, 300
        lang_feat=torch.stack(lang_feat, 0)
        # Obj_num, 
        lang_len = np.array(lang_len).astype(np.int64)
        # Obj_num, 
        lang_objID=torch.LongTensor(lang_objID)
        lang_cls=torch.LongTensor(lang_cls)

        batch_lang_feat.append(lang_feat)
        batch_lang_len.append(lang_len)
        batch_lang_objID.append(lang_objID)
        batch_lang_cls.append(lang_cls) 
        
        a,b,c,d=train_3d[scene_id]
        #print (c.shape)
        #print (d.shape)
        #print (c.dtype)
        #print (d.dtype)
        
        #m=np.eye(3)+np.random.randn(3,3)*0.1
        coord = a
        m=np.eye(3)+np.random.randn(3,3)*0.1
        #m1=np.eye(3)
        m[0][0]*=np.random.randint(0,2)*2-1
        m*=scale
        #m1*=scale1
        theta=np.random.rand()*2*math.pi
        m=np.matmul(m,[[math.cos(theta),math.sin(theta),0],[-math.sin(theta),math.cos(theta),0],[0,0,1]])
        a=np.matmul(a,m)
        #m1=np.matmul(m1,[[math.cos(theta),math.sin(theta),0],[-math.sin(theta),math.cos(theta),0],[0,0,1]])
        #a1=np.matmul(a1,m1)
        
        #a *= scale 
        #coord = a/scale
        if elastic_deformation:
            a=elastic(a,6*scale//50,40*scale/50)
            a=elastic(a,20*scale//50,160*scale/50)
        m=a.min(0)
        M=a.max(0)
        q=M-m
        offset=-m+np.clip(full_scale-M+m-0.001,0,None)*np.random.rand(3)+np.clip(full_scale-M+m+0.001,None,0)*np.random.rand(3)
        a+=offset
        idxs=(a.min(1)>=0)*(a.max(1)<full_scale)
        a=a[idxs]
        b=b[idxs]
        c=c[idxs]
        d=d[idxs]
        coord = coord[idxs]
        coord = -1 + 2*(coord-coord.min(0)[0])/(coord.max(0)[0]-coord.min(0)[0])
        a=torch.from_numpy(a).long()
        locs.append(torch.cat([a,torch.LongTensor(a.shape[0],1).fill_(idx)],1))
        feats.append(torch.from_numpy(b)+torch.randn(3)*0.1)
        #a1=torch.from_numpy(a1).long()
        #locs1.append(torch.cat([a1,torch.LongTensor(a1.shape[0],1).fill_(idx)],1))
        #feats1.append(torch.from_numpy(b))
        labels.append(torch.from_numpy(c))
        ins_labels.append(torch.from_numpy(d.astype(int)-1))
        coords.append(torch.from_numpy(coord))
        num_points.append(a.shape[0])
        scene_names.append(scene_id)

        # Label
        # Num_points, Supvox_num
        spvox, spvox_ins, spvox_sem = supervoxel[scene_id]
        spvox = torch.from_numpy(spvox).long().unsqueeze(-1)
        spvox_ins = torch.from_numpy(spvox_ins).long()
        spvox_sem = torch.from_numpy(spvox_sem).long()
        #ref_lbl = (ins_labels[-1].unsqueeze(-1)-1) == lang_objID
        ref_lbl = spvox_ins.unsqueeze(-1) == lang_objID
        # Supvox_num, Obj_num,
        batch_ref_lbl.append(ref_lbl.long())
        batch_supvox.append(torch.cat([spvox, torch.zeros_like(spvox)+idx],-1))
        batch_supvox_lbl.append([spvox_ins, spvox_sem])
    locs=torch.cat(locs,0)
    #locs1=torch.cat(locs1,0)
    feats=torch.cat(feats,0)
    #feats1=torch.cat(feats1,0)
    labels=torch.cat(labels,0)
    ins_labels=torch.cat(ins_labels,0)
    coords = torch.cat(coords,0)
    batch_supvox = torch.cat(batch_supvox,0)
    batch_data = {'x': [locs,feats],
                  'y': labels.long(),
                  'id': tbl,
                  'y_ins': ins_labels.long(),
                  'coords': coords,
                  'num_points': num_points,
                  'names': scene_names,
                  'lang_feat': batch_lang_feat,
                  'lang_len': batch_lang_len,
                  'lang_objID': batch_lang_objID,
                  'lang_cls': batch_lang_cls,
                  'supvox': [batch_supvox,batch_supvox_lbl],
                  'ref_lbl': batch_ref_lbl} 
    return batch_data

print (len(loader_list))
train_data_loader = torch.utils.data.DataLoader(
    #list(range(len(scanrefer))),
    loader_list, 
    batch_size=batch_size,
    collate_fn=trainMerge,
    num_workers=1, 
    shuffle=True,
    drop_last=True,
    worker_init_fn=lambda x: np.random.seed(x+int(time.time()))
)

#for i, batch in enumerate(train_data_loader):
#	print (i)
	
	
