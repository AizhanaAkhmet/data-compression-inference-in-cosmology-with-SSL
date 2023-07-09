import torch
import torch.nn as nn
from torch.functional import F

## adapted from "generally intellient" team's code 
## https://generallyintelligent.com/open-source/2022-04-21-vicreg/
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def vicreg_loss(z_a, z_b, 
                invariance_loss_weight = 25, 
                variance_loss_weight   = 25,
                covariance_loss_weight = 1):
    assert z_a.shape == z_b.shape and len(z_a.shape) == 2
    # invariance loss
    loss_inv = F.mse_loss(z_a, z_b)
    
    # variance loss
    std_z_a = torch.sqrt(z_a.var(dim=0) + 0.0001) 
    std_z_b = torch.sqrt(z_b.var(dim=0) + 0.0001) 
    loss_v_a = torch.mean(F.relu(1 - std_z_a))
    loss_v_b = torch.mean(F.relu(1 - std_z_b)) 
    loss_var = (loss_v_a + loss_v_b) 
    
    z_a = z_a - z_a.mean(dim=0)
    z_b = z_b - z_b.mean(dim=0)

    # covariance loss
    N, D = z_a.shape
    cov_z_a = ((z_a.T @ z_a) / (N - 1)) # DxD
    cov_z_b = ((z_b.T @ z_b) / (N - 1)) # DxD
    loss_cov = off_diagonal(cov_z_a).pow_(2).sum().div(D) 
    loss_cov += off_diagonal(cov_z_b).pow_(2).sum().div(D)
    
    weighted_inv = loss_inv * invariance_loss_weight
    weighted_var = loss_var * variance_loss_weight
    weighted_cov = loss_cov * covariance_loss_weight

    loss = weighted_inv + weighted_var + weighted_cov   
    return loss, weighted_inv, weighted_var, weighted_cov

def vicreg_loss_pairs(zs_a, zs_b, n_pairs,
                    invariance_loss_weight = 25, 
                    variance_loss_weight = 25,
                    covariance_loss_weight = 1):
    # invariance loss
    losses_inv, losses_cov, losses_var = 0., 0., 0.
    for n in range(n_pairs):
        z_a, z_b = zs_a[n], zs_b[n]
        # invariance loss
        loss_inv = F.mse_loss(z_a, z_b)
        losses_inv += loss_inv
        
        # variance loss
        std_z_a = torch.sqrt(z_a.var(dim=0) + 0.0001) 
        std_z_b = torch.sqrt(z_b.var(dim=0) + 0.0001) 
        loss_v_a = torch.mean(F.relu(1 - std_z_a))
        loss_v_b = torch.mean(F.relu(1 - std_z_b)) 
        loss_var = (loss_v_a + loss_v_b)
        
        losses_var += loss_var
        
        # covariance loss
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)
        
        N, D = z_a.shape
        cov_z_a = ((z_a.T @ z_a) / (N - 1)) # DxD
        cov_z_b = ((z_b.T @ z_b) / (N - 1)) # DxD
        loss_cov = off_diagonal(cov_z_a).pow_(2).sum().div(D) 
        loss_cov += off_diagonal(cov_z_b).pow_(2).sum().div(D)
        
        losses_cov += loss_cov
    
    losses_inv = losses_inv/n_pairs
    losses_cov = losses_cov/n_pairs
    losses_var = losses_var/n_pairs
    
    weighted_inv = losses_inv * invariance_loss_weight
    weighted_var = losses_var * variance_loss_weight
    weighted_cov = losses_cov * covariance_loss_weight

    loss = weighted_inv + weighted_var + weighted_cov   
    return loss, weighted_inv, weighted_var, weighted_cov

def vicreg_loss_zs(zs, 
                   invariance_loss_weight = 25, 
                   variance_loss_weight   = 25,
                   covariance_loss_weight = 1):
    
    loss_inv, loss_cov, loss_var = 0., 0., 0.
    # invariance loss
    for a in range(len(zs)):
        for b in range(a+1, len(zs)):
            z_a = zs[a]
            z_b = zs[b]
            loss_inv += F.mse_loss(z_a, z_b)
            
    
    # variance loss, covariance loss
    for a in range(len(zs)):
        z_a = zs[a]
        std_z_a  = torch.sqrt(z_a.var(dim=0) + 0.0001) 
        loss_v_a = torch.mean(F.relu(1 - std_z_a))
        loss_var += loss_v_a
        
    
        z_a = z_a - z_a.mean(dim=0)
        N, D = z_a.shape
        cov_z_a = ((z_a.T @ z_a) / (N - 1)) # DxD
        loss_cov += off_diagonal(cov_z_a).pow_(2).sum().div(D)
        
    weighted_inv = loss_inv * invariance_loss_weight /len(zs)
    weighted_var = loss_var * variance_loss_weight /len(zs)
    weighted_cov = loss_cov * covariance_loss_weight /len(zs)

    loss = weighted_inv + weighted_var + weighted_cov   
    return loss, weighted_inv, weighted_var, weighted_cov
