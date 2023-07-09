import numpy as np
import torch

# Class to create datasets for supervised learning
# It adds rotations and flips to the 2D maps 
class make_dataset():
    def __init__(self, maps, params, rotations=True):
        self.splits = maps.shape[1]
        self.size_map = maps.shape[-1]
        self.total_sims = maps.shape[0]
        self.total_params = params.shape[-1]
        
        maps = maps.reshape(-1, 1, self.size_map, self.size_map)
        params = params.reshape(-1, self.total_params)
        self.size = maps.shape[0]
        self.x    = maps
        self.y    = params
        self.rotations = rotations

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        maps = self.x[idx]
        
        if self.rotations:
            # choose a rotation angle (0-0, 1-90, 2-180, 3-270)
            # and whether do flipping or not
            rot  = np.random.randint(0,4)
            flip = np.random.randint(0,2)

            # rotate and flip the maps
            maps = torch.rot90(self.x[idx], k=rot, dims=[1,2])
            if flip==1:  maps = torch.flip(maps, dims=[1])

        return maps, self.y[idx]
    
#################################################################################
# Class to create datasets for self-supervised learning
# It adds rotations and flips to the 2D maps 
class make_dataset_VICReg():
    def __init__(self, maps, params, rotations=True, n_views=2):
        self.splits = maps.shape[1]
        self.size_map = maps.shape[-1]
        self.total_sims = maps.shape[0]
        self.total_params = params.shape[-1]
        self.n_views = n_views
        
        self.size = maps.shape[0]
        self.x    = maps
        self.y    = params
        self.rotations = rotations

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # choose two random maps
        choices = self.splits
        indices = np.random.choice(choices, size=self.n_views, replace=False)
        
        maps = self.x[idx, indices]

        if self.rotations:
            # choose a rotation angle (0-0, 1-90, 2-180, 3-270)
            # and whether do flipping or not
            rot  = np.random.randint(0,4,size=self.n_views)
            flip = np.random.randint(0,2,size=self.n_views)

            # rotate and flip the maps
            for j in range(self.n_views):
                maps[j] = torch.rot90(maps[j], k=rot[j], dims=[1,2])
                if flip[j]==1:  maps[j] = torch.flip(maps[j], dims=[1])

        return maps, self.y[idx, :self.n_views]  
    
#################################################################################
def create_datasets(maps, params, train_frac, valid_frac, test_frac, seed = None, VICReg=False, rotations=True, n_views=2):
    assert maps.shape[0] == params.shape[0]
    dset_size = maps.shape[0]
    
    if seed is not None:
        np.random.seed(seed)
    
    # randomly shuffle the simulations 
    sim_numbers = np.arange(dset_size) 
    np.random.shuffle(sim_numbers)
    
    # get indices of shuffled maps
    train_size, valid_size, test_size = int(train_frac*dset_size), int(valid_frac*dset_size), int(test_frac*dset_size)
    train_ind = sim_numbers[:train_size]
    valid_ind = sim_numbers[train_size:(train_size+valid_size)]
    test_ind  = sim_numbers[(train_size+valid_size):]

    maps_train, params_train = maps[train_ind], params[train_ind]
    maps_valid, params_valid = maps[valid_ind], params[valid_ind]
    maps_test, params_test   = maps[test_ind], params[test_ind]
    
    if VICReg:
        train_dset = make_dataset_VICReg(maps_train, params_train, rotations, n_views)
        valid_dset = make_dataset_VICReg(maps_valid, params_valid, rotations, n_views)
        test_dset  = make_dataset_VICReg(maps_test, params_test, rotations, n_views)
    else:
        train_dset = make_dataset(maps_train, params_train, rotations)
        valid_dset = make_dataset(maps_valid, params_valid, rotations)
        test_dset  = make_dataset(maps_test, params_test, rotations)

    return train_dset, valid_dset, test_dset
    