import numpy as np
import torch
from torch.functional import F
import torch.nn as nn

def generate_params(size, 
                    splits, # number of augmentations
                    predict_D = True, # whether B != D
                    seed = None, 
                    kpivot = 0.5,
                    A_min = 0.1, A_max = 1., 
                    B_min = -1, B_max = 0, 
                    C_min = 0.5, C_max = 1.5, 
                    D_min = -0.5, D_max = 0.5,
                    Pk_continuous = True, 
                    only_cosmic_var = False # whether different augmentations have different values of D
                   ):
    if seed is not None:
        np.random.seed(seed)
    
    # get the values of the parameters/labels
    A  = A_min + (A_max - A_min)*np.random.random(size) #(0.9*np.random.random(size)+0.1)
    B  = B_min + (B_max - B_min)*np.random.random(size) #-1.0 + 1.0*np.random.random(size)
    
    A = A[:, None].repeat(splits, axis = 0).flatten()
    B = B[:, None].repeat(splits, axis = 0).flatten()


    params = [A, B]
    if predict_D:
        if only_cosmic_var:
            D  = D_min + (D_max - D_min)*np.random.random(size) #-0.5 + np.random.random(size)
            D  = D[:, None].repeat(splits, axis = 0).flatten()
        else:    
            D  = D_min + (D_max - D_min)*np.random.random(size*splits) #-0.5 + np.random.random(size*splits)
        C  = A*kpivot**(B - D)
        if not Pk_continuous:
            C = alpha*C
        params.append(C)
        params.append(D)
    
    else:
        D = B.copy()
        C = A.copy()
        params.append(C)
        params.append(D)
    
    
    params = np.array(params).T
    return params     
    
def get_Pk_arr(k, Nk, params, predict_D = True, kpivot = 0.5, seed = None):
    num_params = params.shape[-1]
    splits = params.shape[-2]
    params = params.reshape(-1, num_params)
    
    A, B = params[:, 0:1], params[:, 1:2]
    k = k[None, :]
    Pk = A*k**B
    
    # get the hydro Pk part
    if predict_D:
        C, D = params[:, 2:3], params[:, 3:]
        indexes = np.where(k>kpivot)[1]
        
        if len(indexes)>0:
            Pk[:, indexes] = C*k[:, indexes]**D
    
    if seed is not None:
        np.random.seed(seed)
    # add cosmic variance
    dPk = np.sqrt(2*Pk**2/Nk)
    Pk  = np.random.normal(loc=Pk, scale=dPk)
    
    Pk = Pk.reshape(-1, splits, Pk.shape[-1])
    return Pk

####################################################################################
# custom dataset and augmentations classes
class AugmentationTransformations(object):

    def __init__(self, n_views = 2, n_splits = 10): 
        self.n_views = n_views
        self.n_splits = n_splits

    def __call__(self, x):
        indices = np.random.choice(self.n_splits, size=self.n_views, replace=False)
        return [x[i][None, :] for i in indices]
    
class customDataset():
    def __init__(self, data, params, transform = None):
        self.size = data.shape[0]
        self.transform = transform
        self.x = data
        self.y = params[:, 0, :]
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        sample = self.transform(self.x[idx])
        return torch.cat(sample), self.y[idx]
####################################################################################
