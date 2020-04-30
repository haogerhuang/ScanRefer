import torch, os, numpy as np, glob, multiprocessing as mp
from sklearn.cluster import MeanShift as MS
from sklearn.cluster import KMeans

NYU_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

def mkpredictions_test(groups, predictions, scene_name, save_dir):
    assert groups.ndim > 1
    f = open(os.path.join(save_dir, scene_name+'.txt'), 'w')
    mask_dir = os.path.join(save_dir, 'predicted_masks')
    if not os.path.exists(mask_dir):
        os.mkdir(mask_dir)
    num_grp = 0
    if groups.ndim == 1:
        groups = np.expand_dims(groups, -1) == np.unique(groups)
    for i in range(groups.shape[1]):
        if i == -1: continue
        grp = groups[:,i]
        if grp.sum() < 100: continue
        f_name = os.path.join('predicted_masks', scene_name+'_'+str(int(num_grp)).rjust(3, '0')+'.txt')
        np.savetxt(os.path.join(save_dir, f_name), grp, fmt='%i')
        grp_pred = predictions[grp.astype(bool)].mean(0)
        lbl = grp_pred.argmax(0)
        conf = grp_pred[lbl]
        f.write(f_name+' '+str(NYU_CLASS_IDS[lbl])+' '+str(conf)+'\n')
        num_grp += 1

def MS(feat, thres):
	cluste

def MS_v2(embedding, thres):
	grps = torch.zeros(embedding.shape[0])-1
	indices = torch.arange(grps.shape[0])
	if embedding.is_cuda:
		grps = grps.cuda()
		indices = indices.cuda()
	not_assigned = (grps == -1).sum()
	num_grps = 1
	while(not_assigned > 0):
		m = grps == -1
		idx = np.random.choice(indices[m].cpu(), 1)
		diff = float('inf')
		center = embedding[idx]
		while(diff > 1e-6):
			d = (((embedding[m] - center)**2).sum(-1))**0.5
			m1 = d <= thres
			new_center = embedding[m][m1].mean(0)
			diff = ((new_center - center)**2).sum()
			center = new_center
		grps[m] += m1.long()*num_grps
		not_assigned = (grps == -1).sum()
		num_grps += 1
	return grps.long()

def MS_v3(x, thres, samples=2048):
	if x.shape[0] > samples:
		indice = np.random.choice(x.shape[0], samples, replace=False)
		targets = x[indice]
	else:
		targets = x
	results = []
	for t in targets:
		#print (t)
		diff = float('inf')
		center = t
		while(diff > 0):
			d = (((x-center)**2).sum(-1))**0.5
			m = d < thres
			if m.sum() == 0: break
			new_center = x[m].mean(0)
			diff = ((new_center - center)**2).sum()
			center = new_center
		results.append(center)
	results = torch.stack(results, 0)
	results = torch.unique(results,dim=0)
	centers = []
	while(results.shape[0] > 0):
		i = np.random.choice(results.shape[0], 1)
		mask = (((results[i]-results)**2).sum(-1))**0.5 < thres
		centers.append(results[mask].mean(0))
		results = results[mask == False]			
	centers = torch.stack(centers, 0)	
	d = (centers**2).sum(1) + (x**2).sum(1).unsqueeze(-1) - 2*torch.matmul(x, centers.permute(1,0))
	grps = d.argmin(-1)
	grps_ = grps.unsqueeze(-1) == torch.unique(grps)
	score = (d * grps_.float()).sum(0)/grps_.float().sum(0)
	return grps.long(), -score
	#return cluster.labels_

	

def NMS(cluster, score, thres=0.5):
    #cluster N, C
    # C 
    results = []
    #cluster_scores = torch.matmul(cluster.permute(1,0).float(), pred).max(1)[0]
    sort_idx = score.argsort(descending=True)  
    cluster = cluster[:, sort_idx]
    cluster = cluster.cpu()
    while(cluster.shape[-1] > 0):
        #print (cluster.shape) 
        if len(sort_idx) <= 1: break
        target = cluster[:,0]
        results.append(target) 
        target = target.unsqueeze(-1)
        if cluster.shape[1] == 1: break
        cluster = cluster[:, 1:]	
        ious = (target*cluster).float().sum(0)/(target|cluster).float().sum(0)
        m = ious <= thres
        cluster = cluster[:, m]
    results = torch.stack(results, 1)
    return results

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

