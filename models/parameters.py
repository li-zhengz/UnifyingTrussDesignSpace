from cmath import sqrt
import torch
import numpy as np
from numpy import genfromtxt
from scipy import sparse
import pandas as pd
import os
import torch.nn as nn
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def vec2tensor(C_vec):
    """
    Converts a vector representation of a stiffness matrix to a 6x6 tensor.

    Parameters:
    C_vec (array-like): A 1D array of length 9 representing the stiffness matrix.

    Returns:
    np.ndarray: A 6x6 stiffness matrix.
    """
    C_tensor = np.zeros((6, 6))
    C_tensor[0, 0], C_tensor[0, 1], C_tensor[0, 2] = C_vec[0], C_vec[1], C_vec[2]
    C_tensor[1, 0], C_tensor[1, 1], C_tensor[1, 2] = C_vec[1], C_vec[3], C_vec[4]
    C_tensor[2, 0], C_tensor[2, 1], C_tensor[2, 2] = C_vec[2], C_vec[4], C_vec[5]
    C_tensor[3, 3], C_tensor[4, 4], C_tensor[5, 5] = C_vec[6], C_vec[7], C_vec[8]
    return C_tensor

def s2vec(S_matrix):
    """
    Converts a compliance matrix to a vector representation.

    Parameters:
    S_matrix (np.ndarray): A 6x6 compliance matrix.

    Returns:
    np.ndarray: A 1D array of length 12 representing the compliance matrix.
    """
    S_vec = np.zeros(12)
    S_vec[0] = 1. / S_matrix[0, 0]
    S_vec[1] = 1. / S_matrix[1, 1]
    S_vec[2] = 1. / S_matrix[2, 2]
    S_vec[3] = 1. / S_matrix[3, 3]
    S_vec[4] = 1. / S_matrix[4, 4]
    S_vec[5] = 1. / S_matrix[5, 5]
    S_vec[6] = -S_matrix[0, 1] * S_vec[1]
    S_vec[7] = -S_matrix[0, 2] * S_vec[2]
    S_vec[8] = -S_matrix[1, 2] * S_vec[2]
    S_vec[9] = -S_matrix[1, 0] * S_vec[0]
    S_vec[10] = -S_matrix[2, 0] * S_vec[0]
    S_vec[11] = -S_matrix[2, 1] * S_vec[1]
    return S_vec

numNodes = 27
dim = 3

ax_dim = 32
a_dim = 8
x_dim = 8

latent_dim = ax_dim + x_dim + a_dim
z_dim = latent_dim

c_hidden_dim = [latent_dim, 400, 800, 1000, 400, 400, 200]
inv_hidden_dim = [500, 500, 600, 500, 300, 200]

epochs = 200
dropout = 0.
batch_size = 512
test_batch_size = batch_size

kl_update = 'annealing'

valid_size = 0.03

torch.backends.cudnn.benchmark = True

outputFolder = '../results'
invOutputFolder = '../results'
outputFile = outputFolder+'/output.txt'
parameterFile = outputFolder + '/parameters.txt'
savedModelFolder = '../results'
load_pretrained_model = False

c_lr = 1e-3
learningRate = 5e-4
inv_lr = 1e-4

target = 'stiffness'
paramDim = 9
add_noise = False
recon_loss_func = 'l2'

if recon_loss_func == 'l1':
    recon_criterion = nn.L1Loss(reduction = 'sum')
elif recon_loss_func == 'l2':
    recon_criterion = nn.MSELoss(reduction = 'sum')
elif recon_loss_func == 'bce':
    recon_criterion = nn.BCELoss(reduction = 'sum')

folder = '../data'
perturbation = sparse.load_npz(folder+'/node-offset.npz').toarray()
nodes = sparse.load_npz(folder+'/node-position.npz').toarray()
stiffness = torch.tensor((pd.read_csv(folder+'/stiffness-vec.csv', delimiter = ",", header = None)).values).float()
adj_data = sparse.load_npz(folder+'/adjacency-matrix.npz').toarray()

numNodes = adj_data.shape[-1]
numIter = int(adj_data.shape[0]/numNodes)

print('Loading data from ', folder)
print("Number of truss unit cells = ", numIter)

ptb_mask = np.ones([numNodes*dim])
ptb_vec = []
for i in range(numIter):
    ptb_x = perturbation[i*numNodes:(i+1)*numNodes,:].flatten()
    ptb_vec.append(ptb_x)
zero_id = np.where(np.sum(np.array(ptb_vec), axis = 0) == 0.)[0]
ptb_mask[zero_id] = 0.
ptb_mask = ptb_mask.reshape([numNodes,dim])
np.savetxt(folder+'/ptb_mask.csv', ptb_mask, delimiter = ",")

a_row, a_col = np.triu_indices(numNodes)
adj_vec_dim = len(a_row)

x_row, x_col = np.nonzero(ptb_mask)
ptb_vec_dim = len(x_row)

check_file = os.path.exists(folder+'/train_ptb_norm.npz')
if check_file == True:
    adj_list = sparse.load_npz(folder+'/train_adj.npz').toarray()
    x = sparse.load_npz(folder+'/train_ptb.npz').toarray()
    ptb_norm = sparse.load_npz(folder+'/train_ptb_norm.npz').toarray()
else:
    a = []
    total_num_ptb_node = []
    for i in range(numIter):
        x_iterk = perturbation[i*numNodes:(i+1)*numNodes,:].copy()
        xrow, xcol = np.nonzero(x_iterk)
        a.append(x_iterk[x_row,x_col])
        total_num_ptb_node.extend([len(np.unique(np.concatenate((xrow, xcol))))])

    ptb_data = np.array(a)
    ptb_norm = ptb_data.copy()

    adj_list = np.zeros([numIter, len(a_row)])
    x = np.zeros([numIter, ptb_vec_dim])
    for iterk in range(numIter):
        tmp = adj_data[iterk * numNodes : (iterk + 1) * numNodes , :].copy()
        tmp += tmp.transpose()
        tmp [tmp != 0.] = 1.
        adj_list[iterk,:] = tmp[a_row,a_col]
        x[iterk, :] = ptb_norm[iterk , :].copy()

    sparse.save_npz(folder+'/train_ptb.npz', sparse.csr_matrix(x))
    sparse.save_npz(folder+'/train_adj.npz', sparse.csr_matrix(adj_list))
    sparse.save_npz(folder+'/train_ptb_norm.npz', sparse.csr_matrix(ptb_norm))

adj_list = torch.from_numpy(adj_list).float()
x = torch.from_numpy(x).float()
ptb_mask = torch.from_numpy(ptb_mask).float()

# Calculate moduli
check_file = os.path.exists(folder+'/moduli.csv')

if check_file == True:
    pass
else:
    c_data = stiffness.numpy()
    s_name = ['E1', 'E2', 'E3', 'G23', 'G31', 'G12', 'v21', 'v31', 'v32', 'v12', 'v13', 'v23']
    moduli = np.zeros([c_data.shape[0],len(s_name)])
    singular_id = []
    for i in range(c_data.shape[0]):
        c_vec = c_data[i,:]
        c_tensor = vec2tensor(c_vec)
        if np.linalg.det(c_tensor) == 0.:
            singular_id.append(i)
        else:
            s_tensor = np.linalg.inv(c_tensor)
            moduli[i,:] = s2vec(s_tensor)
    np.savetxt(folder+'/moduli.csv', moduli, delimiter = ",")

