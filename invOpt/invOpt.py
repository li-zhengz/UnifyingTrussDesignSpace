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
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, RawArray, Process
import os.path
import pytorchtools

import pickle
torch.autograd.set_detect_anomaly(True)

torch.manual_seed(0)
torch.backends.cudnn.benchmark = True

device = 'cpu'
model = vaeModel()
c_model = c_MLP()

model.load_state_dict(torch.load(outputFolder+'/best_model.pt', map_location=torch.device('cpu')))
c_model.load_state_dict(torch.load(outputFolder+'/best_c_model.pt', map_location=torch.device('cpu')))

model.eval()
c_model.eval()

model.to(device)
c_model.to(device)
ptb_mask = pd.read_csv(folder+'/ptb_mask.csv', delimiter = ",", header = None).to_numpy()

a_row, a_col = np.triu_indices(numNodes)
x_row, x_col = np.nonzero(ptb_mask)

relu_func = nn.ReLU()
# *********************************
# ******** Initialization *********
# *********************************

opt_target = ['E33']
opt_value = [300.]

inverseSaveFolder = outputFolder+'/opt/'
for index, target in enumerate(opt_target):
    if index == len(opt_target)-1:
        inverseSaveFolder += target
    else:
        inverseSaveFolder += target + '_'

dataset  = TensorDataset(adj_list, x, c_data)
num_train = len(dataset)
data_loader = DataLoaderX(dataset, batch_size = num_train, shuffle = False, pin_memory = True)

optimization_method = ['Adam']
opt_epoch = 1200
num_sample = 100
Adam_lr = 1e-4
trace_back = True
recon_criterion = nn.MSELoss(reduction = 'sum')
num_workers = 40

start = time.time()
var_dict = {}

if os.path.exists(outputFolder + "/train_std.csv"):
    train_z = pd.read_csv(outputFolder+"/train_z.csv", delimiter = ",", header = None).to_numpy()
    train_mu = pd.read_csv(outputFolder+"/train_mu.csv", delimiter = ",", header = None).to_numpy()
    train_std = pd.read_csv(outputFolder+"/train_std.csv", delimiter = ",", header = None).to_numpy()
else:
    train_z = []
    train_mu = []
    train_std = []
    for adj, x, c in data_loader:
        adj = adj.to(device)
        x = x.to(device)
        c = c.to(device)
        encoded, mu, std = model.encoder(adj, x)
        train_z.extend(encoded.detach().numpy())
        train_mu.extend(mu.detach().numpy())
        train_std.extend(std.detach().numpy())

    np.savetxt(outputFolder+"/train_z.csv", np.array(train_z), delimiter = ",")
    np.savetxt(outputFolder+"/train_mu.csv", np.array(train_mu), delimiter = ",")
    np.savetxt(outputFolder+"/train_std.csv", np.array(train_std), delimiter = ",")

dtype = torch.float
train_z = np.array(train_z)

print("**************************************")
print("***** Inverse design target ",opt_target[0]," *****")
print("**************************************") 

try:
    os.system('mkdir '+inverseSaveFolder+'/')
except OSError:
    print ("Creation of the directory failed")
else:
    print ('Successfully created the directory ' + inverseSaveFolder)


moduli = pd.read_csv(folder+'/moduli.csv', delimiter = ",", header = None).to_numpy()
loss = 0.

for count, k in enumerate(opt_target):
    target_index = s_name.index(k)
    if opt_value[count] < 0:
        print("Minimum found in dataset: ", k, " ",np.min(moduli[:,target_index]))
    else:
        print("Maximum found in dataset: ", k, " ",np.max(moduli[:,target_index]))
    target_value = opt_value[count]
    target_arr = np.ones([stiffness_vec.shape[0], 1])*target_value
    loss += np.linalg.norm(moduli[:,target_index].reshape([len(moduli),1]) - target_arr ,axis = 1)
    
initial_idx = np.argsort(loss)[:int(num_sample)]
print("Number of initial guesses = ", len(initial_idx))
initial_adj = adj_list[initial_idx,:].detach().clone().float()
initial_x = x[initial_idx,:].detach().clone().float()

initial_z, initial_mu, initial_std = model.encoder(initial_adj, initial_x)
initial_z += torch.randn_like(initial_z)

var_dict = {}
updated_z = np.zeros(initial_z.shape)
predicted_c = np.zeros([len(initial_z), paramDim])
optimization = optimization_method[0]

# ************************************************
# ******** Define Multiprocessing worker *********
# ************************************************

start = time.time()

initial_z.share_memory_()

patience = 200

def init_workder(x, x_shape, y, y_shape, z, z_shape, m, m_shape, n, n_shape):
    var_dict['x'] = x
    var_dict['x_shape'] = x_shape

    var_dict['y'] = y
    var_dict['y_shape'] = y_shape

    var_dict['z'] = z
    var_dict['z_shape'] = z_shape

    var_dict['m'] = m
    var_dict['m_shape'] = m_shape

    var_dict['n'] = n
    var_dict['n_shape'] = n_shape

def worker_func(idx):
    es = 0
    best_acc = 1e10

    updated_z_np = np.frombuffer(var_dict['x']).reshape(var_dict['x_shape'])
    predicted_c_np = np.frombuffer(var_dict['y']).reshape(var_dict['y_shape'])
    z_trace_np = np.frombuffer(var_dict['z']).reshape(var_dict['z_shape'])

    updated_adj_np = np.frombuffer(var_dict['m']).reshape(var_dict['m_shape'])
    updated_x_np = np.frombuffer(var_dict['n']).reshape(var_dict['n_shape'])

    optimization = 'Adam'
    initial_guess_z = (initial_z[idx,:]).reshape([1, initial_z.shape[-1]])
    initial_guess_z = torch.tensor(initial_guess_z.detach().numpy(), device = 'cpu').float().requires_grad_()


    if optimization == 'Adam':
        opt = torch.optim.Adam([initial_guess_z], lr = Adam_lr)

    elif optimization == 'NAdam':
        opt = torch.optim.NAdam([initial_guess_z], lr = Adam_lr)

    for e in range(opt_epoch):

        inv_adj_recon, inv_x_recon = model.decoder(initial_guess_z)
        inv_x_recon = relu_func(inv_x_recon - 5e-3)
        inv_adj_recon = relu_func(inv_adj_recon - 0.1)
        inv_z_recon, inv_mu_recon, inv_std_recon = model.encoder(inv_adj_recon, inv_x_recon)

        inv_c_pred = c_model(inv_mu_recon)
        c_tensor = torch_vec2tensor(inv_c_pred, len(inv_c_pred))
        s_vec = torch_tensor2s(c_tensor)
        loss = 0.
        
        for count, k in enumerate(opt_target):
            target_index = s_name.index(k)
            target_pred = s_vec[:,target_index]
            target_value = opt_value[count]
            loss += recon_criterion(target_pred.reshape([len(inv_c_pred), 1]), torch.ones([len(inv_c_pred),1])*target_value) 

        loss.backward(retain_graph = True)
        opt.step() 
        if loss.item() < best_acc:
            best_acc = loss.item()
            best_z = inv_z_recon.detach().numpy()
            best_c = inv_c_pred.detach().numpy()

            best_adj = inv_adj_recon.detach().numpy()
            best_x = inv_x_recon.detach().numpy()
            es = 0
        else:
            es += 1
            if es > patience:
                break
        z_trace_np[idx,e,:] = inv_z_recon.detach().numpy()
    updated_z_np[idx,:] = best_z
    predicted_c_np[idx,:] = best_c

    updated_adj_np[idx,:] = best_adj
    updated_x_np[idx,:] = best_x

x_shape = (num_sample, initial_z.shape[-1])
x_data = np.zeros(x_shape)
x = RawArray('d', x_shape[0]*x_shape[1])
updated_z_np = np.frombuffer(x).reshape(x_shape)
np.copyto(updated_z_np, x_data)

m_shape = (num_sample, initial_adj.shape[-1])
m_data = np.zeros(m_shape)
m = RawArray('d', m_shape[0]*m_shape[1])
updated_adj_np = np.frombuffer(m).reshape(m_shape)
np.copyto(updated_adj_np, m_data)

n_shape = (num_sample, initial_x.shape[-1])
n_data = np.zeros(n_shape)
n = RawArray('d', n_shape[0]*n_shape[1])
updated_x_np = np.frombuffer(n).reshape(n_shape)
np.copyto(updated_x_np, n_data)

y_shape = (num_sample, paramDim)
y_data = np.zeros(y_shape)
y = RawArray('d', y_shape[0]*y_shape[1])
predicted_c_np = np.frombuffer(y).reshape(y_shape)
np.copyto(predicted_c_np, y_data)

z_shape = (num_sample, opt_epoch, initial_z.shape[-1])
z_data = np.zeros(z_shape)
z = RawArray('d', z_shape[0]*z_shape[1]*z_shape[2])
z_trace_np = np.frombuffer(z).reshape(z_shape)
np.copyto(z_trace_np, z_data)

with Pool(processes = num_workers, initializer = init_workder, initargs=(x, x_shape, y, y_shape, z, z_shape, m, m_shape, n, n_shape)) as pool:
    pool.map(worker_func, np.arange(num_sample))

end = time.time()
print("Multiprocessing time = ", end - start)

initial_z_trace = np.array(z_trace_np)
updated_z = np.array(updated_z_np)
updated_adj = np.array(updated_adj_np)
updated_x = np.array(updated_x_np)
predicted_c = np.array(predicted_c_np)

c_pred = predicted_c
c_tensor = torch_vec2tensor(c_pred, len(c_pred))
s_vec = torch_tensor2s(c_tensor)
loss = 0.

for count, k in enumerate(opt_target):
    target_index = s_name.index(k)
    target_pred = s_vec[:,target_index]
    target_value = opt_value[count]
    c_target = np.ones([len(c_pred),1])*target_value
    loss += np.linalg.norm(target_pred - c_target,axis = 1)

best_inv_pred = np.argmin(loss)

for count, k in enumerate(opt_target):
    print("**************************************")
    print("optimization for ",  k, " using ", optimization)
    print("**************************************")
    target_index = s_name.index(k)
    target_pred = s_vec[:,target_index]
    target_value = opt_value[count]
    print("Prediction: ", target_pred[best_inv_pred].detach().numpy())
    print("Target: ", target_value)
    print("Predicted s: \n")
    print(s_name)
    print(np.round(s_vec[best_inv_pred,:].detach().numpy(),4))

best_pred_z = updated_z[best_inv_pred,:]
print("Predicted best id: ", best_inv_pred)
adj_decoded, x_decoded = model.decoder(torch.tensor(updated_z).float())

np.savetxt(inverseSaveFolder+'/ml_prediction.txt', vec2tensor(predicted_c[best_inv_pred,:]).flatten())
np.savetxt(inverseSaveFolder+'/updated_z.csv', updated_z, delimiter = ",")

z_trace_data = {}
for i in range(initial_z_trace.shape[0]):
    tmp = {str(i): initial_z_trace[i,:,:]}
    z_trace_data.update(tmp)
output_file = open(inverseSaveFolder+'/z_trace.pkl', 'wb')
pickle.dump(z_trace_data, output_file)
output_file.close()
