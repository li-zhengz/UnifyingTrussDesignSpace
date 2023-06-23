import torch
import numpy as np
from numpy import genfromtxt
import time
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.autograd import Variable
import pandas as pd
from scipy import sparse
from sklearn.metrics import r2_score 

import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from models.parameters import *
from models.model import *
from models.utils import *
from tqdm import trange
import networkx as nx
from networkx import connected_components

torch.manual_seed(0)
torch.backends.cudnn.benchmark = True

# device = 'cpu'
model = vaeModel()
c_model = c_MLP()

model.load_state_dict(torch.load(outputFolder+'/best_model.pt', map_location=torch.device('cpu')))
c_model.load_state_dict(torch.load(outputFolder+'/best_c_model.pt', map_location=torch.device('cpu')))

model.eval()
c_model.eval()

model.to(device)
c_model.to(device)

ptb_mask = pd.read_csv(folder+'/ptb_mask.csv', delimiter = ",", header = None).to_numpy()

dataset  = TensorDataset(adj_list, x, c_data)

num_train = len(dataset)         

test_batch_size = 2000
split = test_batch_size
train_dataset, test_dataset = random_split(dataset = dataset, lengths = [num_train - split,split])
train_loader = DataLoaderX(train_dataset, batch_size = batch_size, shuffle = True, pin_memory = True)
test_loader = DataLoaderX(test_dataset, batch_size = test_batch_size, shuffle = True, pin_memory = True)

x_test = []; adj_test = []
x_pred = []; adj_pred = []
c_test_mse = 0.

for adj, x, c in test_loader:

    adj = adj.to(device)
    x = x.to(device)
    c = c.to(device)

    encoded, mu, std = model.encoder(adj, x)
    if add_noise == True:
        c_input = encoded
    else:
        c_input = mu
        
    c_pred = c_model(c_input)

    adj_decoded, x_decoded = model.decoder(encoded)

    c_mse = recon_criterion(c_pred, c)
    c_test_mse += c_mse.item()

    x_test = x.cpu().detach().numpy()
    x_pred = x_decoded.cpu().detach().numpy()

    adj_test = adj.cpu().detach().numpy()
    adj_pred = adj_decoded.cpu().detach().numpy()
        
    c_pred = c_pred.cpu().detach().numpy()
    c_test = c.cpu().detach().numpy()

    z_pred = encoded.cpu().detach().numpy()
    mu_pred = mu.cpu().detach().numpy()
    std_pred = std.cpu().detach().numpy()

    np.savetxt( outputFolder+'/validation/x_test.csv', x_test, delimiter=",")
    np.savetxt( outputFolder+'/validation/x_pred.csv', x_pred, delimiter=",")
    sparse.save_npz(outputFolder +'/validation/adj_test.npz', sparse.csr_matrix(adj_test))
    sparse.save_npz(outputFolder +'/validation/adj_pred.npz', sparse.csr_matrix(adj_pred))
    np.savetxt( outputFolder+'/validation/c_test.csv', c_test, delimiter=",")
    np.savetxt( outputFolder+'/validation/c_pred.csv', c_pred, delimiter=",")
    np.savetxt( outputFolder+'/validation/z_pred.csv', z_pred, delimiter=",")
    np.savetxt( outputFolder+'/validation/mu_pred.csv', mu_pred, delimiter=",")
    np.savetxt( outputFolder+'/validation/std_pred.csv', std_pred, delimiter=",")

c_test = pd.read_csv( outputFolder + '/validation/c_test.csv', delimiter = ",", header = None).to_numpy()
c_pred = pd.read_csv( outputFolder + '/validation/c_pred.csv', delimiter = ",", header = None).to_numpy()

x_test = pd.read_csv( outputFolder + '/validation/x_test.csv', delimiter = ",", header = None).to_numpy()
x_pred = pd.read_csv( outputFolder + '/validation/x_pred.csv', delimiter = ",", header = None).to_numpy()

adj_test = sparse.load_npz( outputFolder + '/validation/adj_test.npz').toarray()
adj_pred = sparse.load_npz( outputFolder + '/validation/adj_pred.npz').toarray()

z_pred = pd.read_csv(outputFolder + '/validation/z_pred.csv', delimiter = ",", header = None).to_numpy()

a_row, a_col = np.triu_indices(numNodes)
adj_pred_array = adj_vec2array(adj_pred, a_row, a_col)
adj_test_array = adj_vec2array(adj_test, a_row, a_col)

adj_accuracy = np.sum(adj_pred_array.flatten() == adj_test_array.flatten())/len(adj_test_array)/numNodes
print("Adjacency matrix accuracy = ", "{:.3f}".format(100*adj_accuracy), "%")

sparse.save_npz(outputFolder +'/validation/adj_test.npz', sparse.csr_matrix(adj_test_array))
sparse.save_npz(outputFolder +'/validation/adj_pred.npz', sparse.csr_matrix(adj_pred_array))

x_row, x_col = np.nonzero(ptb_mask)
x_test_array = x_vec2array(x_test, x_row, x_col)
x_pred_array = x_vec2array(x_pred, x_row, x_col)

sparse.save_npz(outputFolder +'/validation/nodes_test.npz', sparse.csr_matrix(x_test_array))
sparse.save_npz(outputFolder +'/validation/nodes_pred.npz', sparse.csr_matrix(x_pred_array))

xname = ['x', 'y', 'z']
cname = np.array([' C11', ' C12', ' C13', ' C22', ' C23', ' C33', ' C44', ' C55', ' C66'])  

for i in range(3):
    print('R2 score of ', xname[i], ' = ' , "{:.3f}".format(100*r2_score(x_test_array[:,i], x_pred_array[:,i])), "%")

for i in range(len(cname)):
    print('R2 score of ', cname[i], ' = ', "{:.3f}".format(100*r2_score(c_test[:,i], c_pred[:,i])), "%")

num_to_sample = 1000
to_sample = np.random.normal(0,1,size = [num_to_sample, z_pred.shape[1]])
latent_z = torch.from_numpy(to_sample).float().to(device)
adj_decoded, x_decoded = model.decoder(latent_z)

adj_decoded = adj_decoded.detach().cpu().numpy()
x_decoded = x_decoded.detach().cpu().numpy()

adj_recon = adj_vec2array(adj_decoded, a_row, a_col)
x_recon = x_vec2array(x_decoded, x_row, x_col)

connected_part = []

for j in range(num_to_sample):
    ex = adj_recon[j*numNodes:(j+1)*numNodes,:]
    ex_x = x_recon[j*numNodes:(j+1)*numNodes,:]
    ex[np.triu_indices(numNodes)] = 0.
    row = np.nonzero(ex)[0]
    col = np.nonzero(ex)[1]
    g = nx.Graph()
    for i in range(row.shape[0]):
        g.add_edge(row[i], col[i])
    num_connected = len(list(list(connected_components(g))))
    connected_part.append(num_connected)
    
connected_part = np.array(connected_part)

valid_id = np.where((connected_part == 1.))[0]
print("Model in ", outputFolder)
print("Validation score = ", "{:.3f}".format(100*valid_id.shape[0]/num_to_sample), "%")