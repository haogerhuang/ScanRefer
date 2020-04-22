import torch
import numpy as np
from itertools import combinations
import sparseconvnet as scn

def InsLoss(fv, ins_lbl, coords, alpha, weighted_dis=True):
    # (num, g)
    if fv.shape[0] > 200000:
        indices = np.random.choice(fv.shape[0], 200000, replace=False)
        fv = fv[indices]
        ins_lbl = ins_lbl[indices]
        coords = coords[indices]
    num, d = fv.shape
    
    grp_1hot = ins_lbl.unsqueeze(-1) == torch.unique(ins_lbl)
    g = grp_1hot.shape[1]
    # (num, g, 1)
    grp_1hot_ = grp_1hot.long().unsqueeze(-1)
    # (g, 1)
    num_in_grp = grp_1hot_.sum(0)
    # (num, 1, d)
    #vox_f = vox_f.view(num, 1, d)
    # (g, d)
    grp_sum = (grp_1hot_ * fv.view(num, 1, d)).sum(0)
    grp_sum /= (num_in_grp + 1e-6)

    reg_loss = ((grp_sum)**2).sum(1).mean()
    if g > 1:
        pairwise_l2 = (grp_sum**2).sum(1) + (grp_sum**2).sum(1).unsqueeze(-1) - \
                        2*torch.matmul(grp_sum, grp_sum.permute(1,0))
        m = torch.triu(torch.ones(g,g), 1) == 1
        dist_loss = pairwise_l2[m]
        dist_loss = torch.max((alpha - dist_loss), torch.zeros_like(dist_loss))
        if weighted_dis:
            coords_mean = (grp_1hot_ * coords.view(num, 1, 3)).sum(0)
            coords_mean /= (num_in_grp + 1e-6)
            pairwise_dis = (coords_mean**2).sum(1) + (coords_mean**2).sum(1).unsqueeze(-1) - \
                           2*torch.matmul(coords_mean, coords_mean.permute(1,0))
            coords_dis = 1/pairwise_dis[m] + 1e-6
            coords_dis /= coords_dis.sum()
            dist_loss = dist_loss * coords_dis
            dist_loss = dist_loss.sum()
        else:
            dist_loss = dist_loss.mean()
    
    else:
        dist_loss = torch.tensor(0.0).cuda()
    #with torch.no_grad():
    ins_lbl = grp_1hot.float().argmax(1)
    grp_mean = grp_sum[ins_lbl.long()]
    #print (fv.shape, each_mean.shape)
    var_loss = ((fv - grp_mean)**2).sum(1, keepdim=True)
    var_loss = torch.max(var_loss, torch.zeros_like(var_loss))  
    #print (var_loss.shape)
    #print (grp_1hot.shape)
    var_loss = (grp_1hot.long() * var_loss).sum(0).unsqueeze(-1)
    var_loss /= (num_in_grp + 1e-6)
    #var_loss = var_loss.sum()/g
    var_loss = var_loss.mean()
    #total_loss += dist_loss
    #result_vox.append(vox_f)
    return {'var':var_loss, 'dist':dist_loss, 'reg':reg_loss}

def InsSemLoss(fv, ins_lbl, sem_lbl, coords, alpha1, alpha2, alpha3,  weighted_dis=False):
    # (num, g)
    #if fv.shape[0] > 200000:
    #    indices = np.random.choice(fv.shape[0], 200000, replace=False)
    #    fv = fv[indices]
    #    ins_lbl = ins_lbl[indices]
    #    sem_lbl = sem_lbl[indices]
    #    coords = coords[indices]
    num, d = fv.shape
    grp_1hot = ins_lbl.unsqueeze(-1) == torch.unique(ins_lbl)
    ins_lbl = grp_1hot.float().argmax(1)
    #g = grp_1hot.shape[1]
    # (g, 1)
    #num_in_grp = grp_1hot.sum(0).unsqueeze(-1)
    # (num, 1, d)
    #vox_f = vox_f.view(num, 1, d)
    # (g, d)
    #grp_mean = []
    #grp_mean = (grp_1hot.long().unsqueeze(-1) * fv.view(num, 1, d)).sum(0)
    #grp_mean /= (num_in_grp + 1e-6)
    gather_func = scn.InputLayer(1, ins_lbl.max()+1, mode=4)
    grp_mean = gather_func([ins_lbl.unsqueeze(-1), fv])
    grp_idx = grp_mean.get_spatial_locations()[:,0]
    grp_mean = grp_mean.features[grp_idx.argsort()]	
    ins_sem = torch.unique(torch.stack([ins_lbl, sem_lbl], 1), dim=0)[:,1]
    pairwise = torch.Tensor(list(combinations(torch.unique(ins_lbl), 2))).long()
    same_sem = ins_sem[pairwise[:,0]] == ins_sem[pairwise[:,1]]
    diff_sem = ins_sem[pairwise[:,0]] != ins_sem[pairwise[:,1]]
    #dist_loss = ((grp_mean[pairwise[:,0]] - grp_mean[pairwise[:,1]])**2).sum(1)
    dist_loss = torch.norm(grp_mean[pairwise[:,0]] - grp_mean[pairwise[:,1]], p=2, dim=1)
    relate_pos = grp_mean[pairwise[:,0]] - grp_mean[pairwise[:,1]]
  
    #mask = grp_1hot == 0
    #grp_coord = grp_1hot.long().unsqueeze(-1) * coords.view(num, 1, 3)
    #INF = torch.zeros_like(grp_coord)
    #INF[mask] = float('inf')
    #max_grp_coord = grp_coord - INF
    #min_grp_coord = grp_coord + INF
    #cen_grp_coord = (max_grp_coord.max(0)[0] + min_grp_coord.min(0)[0])*0.5
    
    same_sem_loss = torch.max((2*alpha2 - dist_loss[same_sem]), torch.zeros_like(dist_loss[same_sem]))
    diff_sem_loss = torch.max((2*alpha3 - dist_loss[diff_sem]), torch.zeros_like(dist_loss[diff_sem]))
    same_sem_loss = (same_sem_loss**2).mean()
    diff_sem_loss = (diff_sem_loss**2).mean()

    #var_loss = ((fv - grp_mean[ins_lbl.long()])**2).sum(1, keepdim=True)
    var_loss = torch.norm(fv - grp_mean[ins_lbl.long()], p=2, dim=1)
    var_loss = torch.max(var_loss-alpha1, torch.zeros_like(var_loss)) 
    
    var_loss = gather_func([ins_lbl.unsqueeze(-1), (var_loss**2).unsqueeze(-1)])
    var_loss = var_loss.features.mean()
    #print (var_loss.shape)
    #print (grp_1hot.shape)
    #var_loss = (grp_1hot.long() * (var_loss**2).unsqueeze(-1)).sum(0).unsqueeze(-1)
    #var_loss /= (num_in_grp + 1e-6)
    #var_loss = var_loss.sum()/g
    #var_loss = var_loss.mean()


    reg_loss = torch.norm(grp_mean, p=2, dim=1).mean()
    return {'var':var_loss, 'dist':[same_sem_loss, diff_sem_loss], 'reg':reg_loss}



def CenLoss(coords, ins_lbl, pred_cen):
    num = coords.shape[0]
    grp_1hot = ins_lbl.unsqueeze(-1) == torch.unique(ins_lbl)
    ins_lbl = grp_1hot.long().argmax(1)
    g = grp_1hot.shape[1]
    # (num, g, 1)
    grp_1hot_ = grp_1hot.long().unsqueeze(-1)
    # (g, 1)
    num_in_grp = grp_1hot_.sum(0)
    # (num, 1, d)
    #vox_f = vox_f.view(num, 1, d)
    # (g, d)
    grp_sum = (grp_1hot_ * coords.view(num, 1, 3)).sum(0)
    grp_sum /= (num_in_grp + 1e-6)
    coords_mean = grp_sum[ins_lbl]
    cen_loss = (coords_mean - pred_cen)**2
    return cen_loss.sum(1).mean()





