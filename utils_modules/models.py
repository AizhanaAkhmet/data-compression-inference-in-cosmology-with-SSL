import math
import numpy as np
import torch
import torch.nn as nn
from torch.functional import F

class SummaryNet(nn.Module):
    def __init__(self, hidden=5, last_layer=10):
        super().__init__()
        # input: 1x100x100 ---------------> output: hiddenx100x100
        self.conv1 = nn.Conv2d(1, hidden, kernel_size = 3, stride=1, padding=1)
        self.B1 = nn.BatchNorm2d(hidden)
        
        # input: hiddenx100x100 ---------------> output: hiddenx100x100
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size = 3, stride=1, padding=1)
        self.B2 = nn.BatchNorm2d(hidden)
        # input: hiddenx100x100 ---------------> output: hiddenx50x50
        # pool
        
        # input: hiddenx50x50 ---------------> output: 2*hiddenx50x50
        self.conv3 = nn.Conv2d(hidden, 2*hidden, kernel_size = 3, stride=1, padding=1)
        self.B3 = nn.BatchNorm2d(2*hidden)
        
        # input: 2*hiddenx50x50 ---------------> output: 2*hiddenx50x50
        self.conv4 = nn.Conv2d(2*hidden, 2*hidden, kernel_size = 3, stride=1, padding=1)
        self.B4 = nn.BatchNorm2d(2*hidden)
        # input: 2*hiddenx50x50 ---------------> output: 2*hiddenx25x25
        # pool
        
        # input: 2*hiddenx25x25 ---------------> output: 4*hiddenx24x24
        self.conv5 = nn.Conv2d(2*hidden, 4*hidden, kernel_size = 2, stride=1, padding=0)
        self.B5 = nn.BatchNorm2d(4*hidden)
        
        # input: 4*hiddenx24x24 ---------------> output: 4*hiddenx24x24
        self.conv6 = nn.Conv2d(4*hidden, 4*hidden, kernel_size = 3, stride=1, padding=1)
        self.B6 = nn.BatchNorm2d(4*hidden)
        # input: 4*hiddenx24x24 ---------------> output: 4*hiddenx12x12
        # pool
        
        # input: 4*hiddenx12x12 ---------------> output: 8*hiddenx10x10
        self.conv7 = nn.Conv2d(4*hidden, 8*hidden, kernel_size = 3, stride=1, padding=0)
        self.B7 = nn.BatchNorm2d(8*hidden)
        
        # input: 8*hiddenx10x10 ---------------> output: 8*hiddenx8x8
        self.conv8 = nn.Conv2d(8*hidden, 8*hidden, kernel_size = 3, stride=1, padding=0)
        self.B8 = nn.BatchNorm2d(8*hidden)
        # input: 8*hiddenx8*8 ---------------> output: 8*hiddenx4x4
        # pool
        
        # input: 8*hiddenx4x4---------------> output: 8*hiddenx2x2
        self.conv9 = nn.Conv2d(8*hidden, 16*hidden, kernel_size = 3, stride=1, padding=0)
        self.B9 = nn.BatchNorm2d(16*hidden)
        # input: 16*hiddenx2x2 ---------------> output: 16*hiddenx1x1
        # pool
        
        self.maxpool = nn.MaxPool2d(2, 2)
        self.pool = nn.AvgPool2d(2, 2) #nn.MaxPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(6, 6)
        # input: hiddenx16x16 ---------------> output: last_layer
        self.fc1 = nn.Linear(16*hidden, 16*hidden)
        self.fc2 = nn.Linear(16*hidden, last_layer)
        
    def forward(self, x):
        x = F.relu(self.B1(self.conv1(x)))
        x = self.pool(F.relu(self.B2(self.conv2(x))))
        
        x = F.relu(self.B3(self.conv3(x)))
        x = self.pool(F.relu(self.B4(self.conv4(x))))
        
        x = F.relu(self.B5(self.conv5(x)))
        x = self.pool(F.relu(self.B6(self.conv6(x))))
        
        x = F.relu(self.B7(self.conv7(x)))
        x = self.pool(F.relu(self.B8(self.conv8(x))))
        x = self.pool(F.relu(self.B9(self.conv9(x))))
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
###############################
def Expander(mlp_units, representation_len, bn = False):
    num_inn_layers = len(mlp_units) - 1
    num_units = [representation_len] + mlp_units
    
    layers = []
    for i in range(num_inn_layers):
        layers.append(nn.Linear(num_units[i], num_units[i + 1]))
        if bn:
            layers.append(nn.BatchNorm1d(num_units[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(num_units[-2], num_units[-1], bias=False))
    return nn.Sequential(*layers)

###############################
def ExpanderSimCLR(mlp_units, representation_dim, output_dim, bn = False):
    layers = []
    if len(mlp_units) == 0:
        layers.append(nn.Linear(representation_dim, output_dim, bias=False))
    else:
        for i in range(len(mlp_units)):
            if i == 0:
                layers.append(nn.Linear(representation_dim, mlp_units[i]))
            else:
                layers.append(nn.Linear(mlp_units[i-1], mlp_units[i]))
            
            if bn:
                layers.append(nn.BatchNorm1d(mlp_units[i]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(mlp_units[-1], output_dim, bias=False))
    return nn.Sequential(*layers) 
    

########################################################################
def vector_to_Cov(vec):
    """ Convert unconstrained vector into a positive-diagonal, symmetric covariance matrix
        by converting to cholesky matrix, then doing Cov = L @ L^T 
        (https://en.wikipedia.org/wiki/Cholesky_decomposition)
    """
    
    D = int((-1.0 + math.sqrt(1.0 + 8.0 * vec.shape[-1])) / 2.0)  # Infer dimensionality; D * (D + 1) / 2 = n_tril
    B = vec.shape[0]  # Batch dim
    
    # Get indices of lower-triangular matrix to fill
    tril_indices = torch.tril_indices(row=D, col=D, offset=0)
    
    # Fill lower-triangular Cholesky matrix
    L = torch.zeros((B, D, D))
    
    mask1 = torch.zeros(L.shape, device=L.device, dtype=torch.bool)
    mask1[:, tril_indices[0], tril_indices[1]] = True
    L = L.masked_scatter(mask1, vec)
    
    # Enforce positive diagonals
    positive_diags = nn.Softplus()(torch.diagonal(L, dim1=-1, dim2=-2))
    
    mask2 = torch.zeros(L.shape, device=L.device, dtype=torch.bool)
    mask2[:, range(L.shape[-1]), range(L.shape[-2])] = True
    L = L.masked_scatter(mask2, positive_diags)
    
    # Cov = L @ L^T 
    Cov = torch.einsum("bij, bkj ->bik",L, L)

    return Cov
########################################################################
class Net(nn.Module):
    def __init__(self, num_classes, hidden = 5):
        super().__init__()
        self.n_params = num_classes  # Number of parameters
        self.n_tril = int(self.n_params * (self.n_params + 1) / 2)  # Number of parameters in lower triangular matrix, for symmetric matrix
        self.n_out = self.n_params + self.n_tril  # Dummy output of neural network
        
        # input: 1x100x100 ---------------> output: hiddenx100x100
        self.conv1 = nn.Conv2d(1, hidden, kernel_size = 3, stride=1, padding=1)
        self.B1 = nn.BatchNorm2d(hidden)
        
        # input: hiddenx100x100 ---------------> output: hiddenx100x100
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size = 3, stride=1, padding=1)
        self.B2 = nn.BatchNorm2d(hidden)
        # input: hiddenx100x100 ---------------> output: hiddenx50x50
        # pool
        
        # input: hiddenx50x50 ---------------> output: 2*hiddenx50x50
        self.conv3 = nn.Conv2d(hidden, 2*hidden, kernel_size = 3, stride=1, padding=1)
        self.B3 = nn.BatchNorm2d(2*hidden)
        
        # input: 2*hiddenx50x50 ---------------> output: 2*hiddenx50x50
        self.conv4 = nn.Conv2d(2*hidden, 2*hidden, kernel_size = 3, stride=1, padding=1)
        self.B4 = nn.BatchNorm2d(2*hidden)
        # input: 2*hiddenx50x50 ---------------> output: 2*hiddenx25x25
        # pool
        
        # input: 2*hiddenx25x25 ---------------> output: 4*hiddenx24x24
        self.conv5 = nn.Conv2d(2*hidden, 4*hidden, kernel_size = 2, stride=1, padding=0)
        self.B5 = nn.BatchNorm2d(4*hidden)
        
        # input: 4*hiddenx24x24 ---------------> output: 4*hiddenx24x24
        self.conv6 = nn.Conv2d(4*hidden, 4*hidden, kernel_size = 3, stride=1, padding=1)
        self.B6 = nn.BatchNorm2d(4*hidden)
        # input: 4*hiddenx24x24 ---------------> output: 4*hiddenx12x12
        # pool
        
        # input: 4*hiddenx12x12 ---------------> output: 8*hiddenx10x10
        self.conv7 = nn.Conv2d(4*hidden, 8*hidden, kernel_size = 3, stride=1, padding=0)
        self.B7 = nn.BatchNorm2d(8*hidden)
        
        # input: 8*hiddenx10x10 ---------------> output: 8*hiddenx8x8
        self.conv8 = nn.Conv2d(8*hidden, 8*hidden, kernel_size = 3, stride=1, padding=0)
        self.B8 = nn.BatchNorm2d(8*hidden)
        # input: 8*hiddenx8*8 ---------------> output: 8*hiddenx4x4
        # pool
        
        # input: 8*hiddenx4x4---------------> output: 8*hiddenx2x2
        self.conv9 = nn.Conv2d(8*hidden, 16*hidden, kernel_size = 3, stride=1, padding=0)
        self.B9 = nn.BatchNorm2d(16*hidden)
        # input: 16*hiddenx2x2 ---------------> output: 16*hiddenx1x1
        # pool
        
        self.maxpool = nn.MaxPool2d(2, 2)
        self.pool = nn.AvgPool2d(2, 2) #nn.MaxPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(6, 6)
        # input: hiddenx16x16 ---------------> output: 128
        self.fc1 = nn.Linear(16*hidden, 16*hidden)
        self.fc2 = nn.Linear(16*hidden, self.n_out)
        
            
    def forward(self, x):
        x = F.relu(self.B1(self.conv1(x)))
        x = self.pool(F.relu(self.B2(self.conv2(x))))
        
        x = F.relu(self.B3(self.conv3(x)))
        x = self.pool(F.relu(self.B4(self.conv4(x))))
        
        x = F.relu(self.B5(self.conv5(x)))
        x = self.pool(F.relu(self.B6(self.conv6(x))))
        
        x = F.relu(self.B7(self.conv7(x)))
        x = self.pool(F.relu(self.B8(self.conv8(x))))
        x = self.pool(F.relu(self.B9(self.conv9(x))))
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

########################################################################
class NetEquivalent(nn.Module):
    def __init__(self, num_classes, hidden = 5):
        super().__init__()
        self.n_params = num_classes  # Number of parameters
        self.n_tril = int(self.n_params * (self.n_params + 1) / 2)  # Number of parameters in lower triangular matrix, for symmetric matrix
        self.n_out = self.n_params + self.n_tril  # Dummy output of neural network
        
        # input: 1x100x100 ---------------> output: hiddenx100x100
        self.conv1 = nn.Conv2d(1, hidden, kernel_size = 3, stride=1, padding=1)
        self.B1 = nn.BatchNorm2d(hidden)
        
        # input: hiddenx100x100 ---------------> output: hiddenx100x100
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size = 3, stride=1, padding=1)
        self.B2 = nn.BatchNorm2d(hidden)
        # input: hiddenx100x100 ---------------> output: hiddenx50x50
        # pool
        
        # input: hiddenx50x50 ---------------> output: 2*hiddenx50x50
        self.conv3 = nn.Conv2d(hidden, 2*hidden, kernel_size = 3, stride=1, padding=1)
        self.B3 = nn.BatchNorm2d(2*hidden)
        
        # input: 2*hiddenx50x50 ---------------> output: 2*hiddenx50x50
        self.conv4 = nn.Conv2d(2*hidden, 2*hidden, kernel_size = 3, stride=1, padding=1)
        self.B4 = nn.BatchNorm2d(2*hidden)
        # input: 2*hiddenx50x50 ---------------> output: 2*hiddenx25x25
        # pool
        
        # input: 2*hiddenx25x25 ---------------> output: 4*hiddenx24x24
        self.conv5 = nn.Conv2d(2*hidden, 4*hidden, kernel_size = 2, stride=1, padding=0)
        self.B5 = nn.BatchNorm2d(4*hidden)
        
        # input: 4*hiddenx24x24 ---------------> output: 4*hiddenx24x24
        self.conv6 = nn.Conv2d(4*hidden, 4*hidden, kernel_size = 3, stride=1, padding=1)
        self.B6 = nn.BatchNorm2d(4*hidden)
        # input: 4*hiddenx24x24 ---------------> output: 4*hiddenx12x12
        # pool
        
        # input: 4*hiddenx12x12 ---------------> output: 8*hiddenx10x10
        self.conv7 = nn.Conv2d(4*hidden, 8*hidden, kernel_size = 3, stride=1, padding=0)
        self.B7 = nn.BatchNorm2d(8*hidden)
        
        # input: 8*hiddenx10x10 ---------------> output: 8*hiddenx8x8
        self.conv8 = nn.Conv2d(8*hidden, 8*hidden, kernel_size = 3, stride=1, padding=0)
        self.B8 = nn.BatchNorm2d(8*hidden)
        # input: 8*hiddenx8*8 ---------------> output: 8*hiddenx4x4
        # pool
        
        # input: 8*hiddenx4x4---------------> output: 8*hiddenx2x2
        self.conv9 = nn.Conv2d(8*hidden, 16*hidden, kernel_size = 3, stride=1, padding=0)
        self.B9 = nn.BatchNorm2d(16*hidden)
        # input: 16*hiddenx2x2 ---------------> output: 16*hiddenx1x1
        # pool
        
        self.maxpool = nn.MaxPool2d(2, 2)
        self.pool = nn.AvgPool2d(2, 2) #nn.MaxPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(6, 6)
        # input: hiddenx16x16 ---------------> output: 128
        self.fc1 = nn.Linear(16*hidden, 16*hidden)
        self.fc2 = nn.Linear(16*hidden, 16*hidden)
        self.fc3 = nn.Linear(16*hidden, 16*hidden)
        self.fc4 = nn.Linear(16*hidden, self.n_out)
        
            
    def forward(self, x):
        x = F.relu(self.B1(self.conv1(x)))
        x = self.pool(F.relu(self.B2(self.conv2(x))))
        
        x = F.relu(self.B3(self.conv3(x)))
        x = self.pool(F.relu(self.B4(self.conv4(x))))
        
        x = F.relu(self.B5(self.conv5(x)))
        x = self.pool(F.relu(self.B6(self.conv6(x))))
        
        x = F.relu(self.B7(self.conv7(x)))
        x = self.pool(F.relu(self.B8(self.conv8(x))))
        x = self.pool(F.relu(self.B9(self.conv9(x))))
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x