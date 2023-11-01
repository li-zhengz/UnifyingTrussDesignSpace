
from itertools import combinations as comb
import numpy as np
from scipy import spatial
import random
import networkx as nx
from networkx import connected_components
# write .scad file
import os
import sys
import math
from tqdm import trange
from moveNodes import *
from scipy import sparse
from numpy import genfromtxt
import torch
from multiprocessing import Pool, RawArray
import time

import scipy

# ***** parameters *****

faceNode = None;
mirror = 1
split = None
dim = 3

nodesInit = np.genfromtxt('nodesInit.csv', delimiter=",")
# conn_list = np.genfromtxt('new_conn_list.csv', delimiter = ",").astype(int)
numNodes = nodesInit.shape[0]

xC = nodesInit[:,0].copy()
yC = nodesInit[:,1].copy()
zC = nodesInit[:,2].copy()
points = np.c_[xC.ravel(), yC.ravel(), zC.ravel()]
tree = spatial.KDTree(points)
rmin = np.sqrt(3)/2.+1e-6
numIter = 20000
writePlot = None
move = 0


A_data = np.zeros([numNodes, numNodes,4])

model = 'bcc'
# temp = np.genfromtxt('baseLattices/' + model + '/A.csv', delimiter=","); A_data[:,:,0] = temp.copy()
temp = np.genfromtxt('baseLattices/' + model + '/A_S.csv', delimiter=","); A_data[:,:,0] = temp.copy()

model = 'sc'
temp = np.genfromtxt('baseLattices/' + model + '/A.csv', delimiter=","); A_data[:,:,1] = temp.copy()
temp = np.genfromtxt('baseLattices/' + model + '/A_S.csv', delimiter=","); A_data[:,:,2] = temp.copy()

model = 'fcc'
# temp = np.genfromtxt('baseLattices/' + model + '/A.csv', delimiter=","); A_data[:,:,4] = temp.copy()
temp = np.genfromtxt('baseLattices/' + model + '/A_S.csv', delimiter=","); A_data[:,:,3] = temp.copy()

baseTop = None

baseLattice = ['sc', 'bcc', 'fcc']
baseS = ['', '_S']

# lattices = ['A1' , 'A2', 'B1', 'B2', 'C2']
lattices = ['A1', 'A2', 'B2', 'C2']
uni = []
for i in range(1000):
    combination = np.random.choice(lattices, 2)
    uni.append(combination[0]+combination[1])
tmp = np.unique(np.array(uni))
uni_copy = tmp.copy()
to_remove = []
for i in tmp:
    for j in tmp:
        if i != j:
            if (i[0] == j[2]) and (i[1] == j[-1]) and (i[2] == j[0]) and(i[-1] == j[1]):
                idx = np.where(tmp== j)
                if i in to_remove:
                    pass
                else:
                    to_remove.append(j)
                    tmp = np.delete(tmp, idx)
                    pass
combination = np.delete(tmp, 1)                    
sumLength = []
stackLattice = []

def moveTop(adj, x, uc_idx):
    if uc_idx == 0:
        to_move_id = np.array([17])
    elif uc_idx == 1:
        to_move_id = np.array([1, 3, 5, 7, 9, 11, 17, 15, 19, 21, 23, 25])
    elif uc_idx == 2:
        to_move_id = np.array([17, 21, 25])
    elif uc_idx == 3:
        to_move_id = np.array([14, 10, 22, 16, 12, 4])
    m_adj = adj.copy()
    m_x = x.copy()

    modify_uc = 1
    if modify_uc == 1:
        for k in to_move_id:
            a = tree.query_ball_point(nodesInit[k,:], rmin) # based on nodes before/after perturbation?
            neighbor_idx = np.array(a)   
            # neighbor_idx = conn_list[k,np.nonzero(conn_list[k,:])[0]]
            # print(neighbor_idx)
            to_move = np.random.choice([0,1],1)
            if to_move == 1:
                m_adj[k,:] = 0.; m_adj[:,k] = 0.
                selected_id = np.random.choice(neighbor_idx,1)
                m_adj[k,selected_id] = 1.
                m_adj[selected_id,k] = 1.
        invalid_node_id = np.where(np.sum(m_adj,axis = 0)==1.)[0]
        new_ex_adj = m_adj.copy()
        iterk = 0
        while iterk >= 0:
            for j in invalid_node_id:
                a = tree.query_ball_point(nodesInit[j,:], rmin)
                neighbor_idx = np.array(a) 
                # neighbor_idx = conn_list[k,np.nonzero(conn_list[k,:])[0]]
                selected_id = np.random.choice(neighbor_idx,1)
                to_move = np.random.choice([0,1],1)
                if to_move == 1:
                    new_ex_adj[j,selected_id] = 1.
                    new_ex_adj[selected_id,j] = 1.
                else:
                    new_ex_adj[j,:] = 0.
                    new_ex_adj[:,j] = 0.
                for k in range(numNodes):
                    new_ex_adj[k,k] = 0.
            invalid_node_id = np.where(np.sum(new_ex_adj,axis = 0) ==1.)[0]
            if invalid_node_id.shape[0]>0:
                iterk += 1
            else:
                break
    else:
        new_ex_adj = m_adj
    return new_ex_adj


def generateUC(choice):

    # move_connection = np.random.choice([True, False], 1)[0]
    move_connection = True

    label = np.where(np.array(lattices) == choice)[0]
    stack = []; s = []; idx = []

    idx, stack, s = str2stru(choice[0]+choice[1])
    
    subA = A_data[:,:,idx] 
    nodes_before_move = nodesInit.copy()
    
    if move_connection == True:
   
        new_adj_0 = moveTop(A_data[:,:,idx], nodesInit, idx)
        A = new_adj_0.copy()

    else:
        A = subA

    nodes_iter = nodesInit.copy()

    return A, nodes_iter, label

def init_workder(x, x_shape, y, y_shape, z, z_shape):
    global var_dict
    var_dict = {}
    var_dict['x'] = x
    var_dict['x_shape'] = x_shape
    var_dict['y'] = y
    var_dict['y_shape'] = y_shape
    var_dict['z'] = z
    var_dict['z_shape'] = z_shape

def worker_func(i):
    x_np = np.frombuffer(var_dict['x']).reshape(var_dict['x_shape'])
    y_np = np.frombuffer(var_dict['y']).reshape(var_dict['y_shape'])
    z_np = np.frombuffer(var_dict['z']).reshape(var_dict['z_shape'])
    choice = random.choices(lattices, k = 1)[0]

    A, nodes_iter, label = generateUC(choice)

    x_np[i*numNodes:(i+1)*numNodes] = A
    y_np[i*numNodes:(i+1)*numNodes] = nodes_iter
    z_np[i, 0] = label

x_shape = (numIter*numNodes,numNodes)
y_shape = (numIter*numNodes,dim)
z_shape = (numIter, 1)
x = RawArray('d', x_shape[0]*x_shape[1])
y = RawArray('d', y_shape[0]*y_shape[1])
z = RawArray('d', z_shape[0]*z_shape[1])
x_data = np.zeros(x_shape)
y_data = np.zeros(y_shape)
z_data = np.zeros(z_shape)
x_np = np.frombuffer(x).reshape(x_shape)
y_np = np.frombuffer(y).reshape(y_shape)
z_np = np.frombuffer(z).reshape(z_shape)
np.copyto(x_np, x_data)
np.copyto(y_np, y_data)
np.copyto(z_np, z_data)
start = time.time()
with Pool(processes = 16, initializer= init_workder, initargs=(x, x_shape, y, y_shape, z, z_shape)) as pool:
    pool.map(worker_func, range(numIter))
end = time.time()
print("time = ", end-start)

nodes0 = y_np.copy()
A0 = x_np.copy()
label0 = np.array(z_np)
ptb0 = nodes0.copy()

for i in range(label0.shape[0]):
    ptb0[i*numNodes:(i+1)*numNodes, :] = ptb0[i*numNodes:(i+1)*numNodes, :] - nodesInit
invalid_id = np.where(np.abs(ptb0)>0.25)[0]
invalid_uc = []
for i in invalid_id:
    invalid_uc.append(int(np.floor(i/numNodes)))
file_invalid_id = np.unique(np.array(invalid_uc))

if file_invalid_id.shape[0] > 0:
    label = np.delete(label0, file_invalid_id, axis = 0)
    invalid_row = []
    for j in file_invalid_id:
        invalid_row_tmp = np.arange(j*numNodes, (j+1)*numNodes)
        invalid_row.extend(invalid_row_tmp)
    invalid_row = np.array(invalid_row)
    ptb = np.delete(ptb0, invalid_row, axis = 0)
    nodes = np.delete(nodes0, invalid_row, axis = 0)
    A = np.delete(A0, invalid_row, axis = 0)
else:
    ptb = ptb0.copy()
    nodes = nodes0.copy()
    A = A0.copy()


def check_bc_connected(adj, x, numUC):
    not_on_boundary = []
    for i in range(numUC):
        ex_adj = adj[i*numNodes:(i+1)*numNodes,:]
        ex_x = x[i*numNodes:(i+1)*numNodes,:]
        ex_adj += ex_adj.transpose()

        row = np.nonzero(ex_adj)[0]
        col = np.nonzero(ex_adj)[0]
        #   max_y = 1.
        bc_i = np.max(ex_x[row,1])
        bc_j = np.max(ex_x[col,1])
        if (bc_i < 1.) and (bc_j < 1.):
            not_on_boundary.append(i)
        #   min_y = 0.
        bc_i = np.min(ex_x[row,1])
        bc_j = np.min(ex_x[col,1])
        if (bc_i > 0.) and (bc_j > 0.):
            not_on_boundary.append(i) 
        #   min_z = 0.
        bc_i = np.min(ex_x[row,-1])
        bc_j = np.min(ex_x[col,-1])
        if (bc_i > 0.) and (bc_j > 0.):
            not_on_boundary.append(i)
        #   max_z = 1.
        bc_i = np.max(ex_x[row,-1])
        bc_j = np.max(ex_x[col,-1])
        if (bc_i < 1.) and (bc_j < 1.):
            not_on_boundary.append(i)
        #   max_x = 1.
        bc_i = np.max(ex_x[row,0])
        bc_j = np.max(ex_x[col,0])
        if (bc_i < 1.) and (bc_j < 1.):
            not_on_boundary.append(i)
        #   min_x = 0.
        bc_i = np.min(ex_x[row,0])
        bc_j = np.min(ex_x[col,0])
        if (bc_i > 0.) and (bc_j > 0.):
            not_on_boundary.append(i)
    return np.array(not_on_boundary)

def check_connection(adj, x, numUC):
    for i in range(numUC):
        ex_adj = adj[i*numNodes:(i+1)*numNodes,:]
        ex_x = x[i*numNodes:(i+1)*numNodes,:].copy()
        ex_adj += ex_adj.transpose()

        invalid_node = np.where(np.sum(ex_adj, axis = 0) == 1.)[0]

        for k in invalid_node:
            if k == 0:
                if ((ex_x[k,0] == 0.) | (ex_x[k,1] == 0.) | (ex_x[k,2] == 0.)) is True:
                    pass
                else:
                    ex_x[k,np.random.choice([0,1,2],1)] = 0.
            elif k == 2:
                if ((ex_x[k,0] == 1.) | (ex_x[k,1] == 0.) | (ex_x[k,2] == 0.)) is True:
                    pass
                else:
                    idx = np.random.choice([0,1,2],1)
                    if idx == 0:  ex_x[k, idx] = 1.
                    else: ex_x[k, idx] = 0.    
            elif k == 6:
                if ((ex_x[k,0] == 0.) | (ex_x[k,1] == 1.) | (ex_x[k,2] == 0.)) is True:
                    pass
                else:
                    idx = np.random.choice([0,1,2],1)
                    if idx == 1: ex_x[k, idx] = 1.
                    else: ex_x[k, idx] = 0.
            elif k == 8:
                if ((ex_x[k,0] == 1.) | (ex_x[k,1] == 1.) | (ex_x[k,2] == 0.)) is True:
                    pass
                else:
                    idx = np.random.choice([0,1,2],1)
                    if idx == 2: ex_x[k, idx] = 0.
                    else: ex_x[k, idx] = 1.
            elif k == 26:
                if ((ex_x[k,0] == 0.) | (ex_x[k,1] == 0.) | (ex_x[k,2] == 1.)) is True:
                    pass
                else:
                    idx = np.random.choice([0,1,2],1)
                    if idx == 2: ex_x[k, idx] = 1.
                    else: ex_x[k, idx] = 0.
            elif k == 28:
                if ((ex_x[k,0] == 1.) | (ex_x[k,1] == 0.) | (ex_x[k,2] == 1.)) is True:
                    pass
                else:
                    idx = np.random.choice([0,1,2],1)
                    if idx == 1: ex_x[k, idx] = 0.
                    else: ex_x[k, idx] = 1.
            elif k == 32:
                if ((ex_x[k,0] == 0.) | (ex_x[k,1] == 1.) | (ex_x[k,2] == 1.)) is True:
                    pass
                else:
                    idx = np.random.choice([0,1,2],1)
                    if idx == 0: ex_x[k, idx] = 0.
                    else: ex_x[k, idx] = 1.
            elif k == 34:
                if ((ex_x[k,0] == 1.) | (ex_x[k,1] == 1.) | (ex_x[k,2] == 1.)) is True:
                    pass
                else:
                   ex_x[k,np.random.choice([0,1,2],1)] = 1.
        x[i*numNodes:(i+1)*numNodes,:] = ex_x.copy()

    return adj, x


tmp = []
new_base_lattice = np.array(A)
numUC = int(new_base_lattice.shape[0]/numNodes)
new_base_lattice, nodes = check_connection(new_base_lattice, nodes, numUC)
for j in range(numUC):
    ex = new_base_lattice[j*numNodes:(j+1)*numNodes,:].copy()
    ex[np.triu_indices(numNodes)] = 0.
    row = np.nonzero(ex)[0]
    col = np.nonzero(ex)[1]
    g = nx.Graph()
    for i in range(row.shape[0]):
        g.add_edge(row[i], col[i])
    tmp.append(len(list(list(connected_components(g)))))
connected_part = np.array(tmp)
non_connected_id = np.where(connected_part != 1.)[0]

invalid_row = []
for j in non_connected_id:
    invalid_row_tmp = np.arange(j*numNodes, (j+1)*numNodes)
    invalid_row.extend(invalid_row_tmp)
invalid_row = np.array(invalid_row)
if invalid_row.shape[0] > 0:
    valid_new_adj = np.delete(new_base_lattice, invalid_row, axis = 0)
    valid_new_nodes = np.delete(nodes, invalid_row, axis = 0)
else:
    valid_new_adj = new_base_lattice.copy()
    valid_new_nodes = nodes.copy()


numUC = int(valid_new_adj.shape[0]/numNodes)
not_on_boundary = check_bc_connected(valid_new_adj, valid_new_nodes, numUC)

file_invalid_id = not_on_boundary
if file_invalid_id.shape[0] > 0:
    invalid_row = []
    for j in file_invalid_id:
        invalid_row_tmp = np.arange(j*numNodes, (j+1)*numNodes)
        invalid_row.extend(invalid_row_tmp)
    invalid_row = np.array(invalid_row)
    new_top_adj = np.delete(valid_new_adj, invalid_row, axis = 0)
    new_top_nodes = np.delete(valid_new_nodes, invalid_row, axis = 0)
else:
    new_top_adj = valid_new_adj.copy()
    new_top_nodes = valid_new_nodes.copy()

numUC = int(new_top_adj.shape[0]/numNodes)
not_on_boundary = check_bc_connected(new_top_adj, new_top_nodes, numUC)

if not_on_boundary.shape[0] > 0:
    print("There are invalid unit cells.")
else:
    print("number of unit cells = ", numUC)
    ptb = np.zeros(new_top_nodes.shape)
    for i in range(numUC):
        tmp = new_top_nodes[i*numNodes:(i+1)*numNodes,:] - nodesInit
        ptb[i*numNodes:(i+1)*numNodes,:] = tmp
        adj_tmp = new_top_adj[i*numNodes:(i+1)*numNodes,:]
        adj_tmp += adj_tmp.transpose()
        valence = np.sum(adj_tmp, axis = 0)
        zero_row = np.where(valence == 0.)[0]
        tmp[zero_row, :] = 0.
        ptb[i*numNodes:(i+1)*numNodes,:] = tmp
    # row, col = np.where((np.abs(ptb)!=0.)&(np.abs(ptb)<0.05))
    # ptb[row, col] = 0.
    for i in range(numUC):
        tmp = ptb[i*numNodes:(i+1)*numNodes,:] + nodesInit
        new_top_nodes[i*numNodes:(i+1)*numNodes,:] = tmp
    # adj_conn_list = np.zeros([new_top_adj.shape[0], 15])
    # triu_idx = torch.triu_indices(numNodes, numNodes).numpy()
    # for i in range(numUC):
    #     ex_adj = new_top_adj[i*numNodes:(i+1)*numNodes,:]
    #     ex_adj[triu_idx[0], triu_idx[1]] = 0.
    #     tmp_adj = ex_adj.transpose()
        # for j in range(numNodes):
        #     act_list = np.nonzero(tmp_adj[j,:])[0]
        #     for k in act_list:
        #         act_id = np.where(k == conn_list[j,:])[0]
        #         adj_conn_list[i*numNodes+j, act_id] = 1.
    # np.savetxt('../../dataSet/'+str(numIter)+'/nodes.csv', new_top_nodes, delimiter=",")
    # np.savetxt('../../dataSet/'+str(numIter)+'/perturbation.csv', ptb, delimiter=",")
    # sparse.save_npz('../../dataSet/'+str(numIter)+'/adj.npz', sparse.csr_matrix(new_top_adj))

    adj_vec = np.zeros([numUC,378])
    for i in range(numUC):
        adj_iter = new_top_adj[i*numNodes:(i+1)*numNodes,:].copy()
        adj_vec[i,:] = adj_iter[np.triu_indices(adj_iter.shape[0])].flatten()
    
    uni_adj, save_id = np.unique(adj_vec, return_index=True, axis=0)
    print("number of unique topologies = ", len(save_id))

    save_ptb = np.zeros([len(save_id)*numNodes,dim])
    save_nodes = np.zeros([len(save_id)*numNodes,dim])
    save_adj = np.zeros([len(save_id)*numNodes,numNodes])

    for i in range(len(save_id)):
        j = save_id[i]
        save_ptb[i*numNodes:(i+1)*numNodes,:] = ptb[j*numNodes:(j+1)*numNodes,:]
        save_nodes[i*numNodes:(i+1)*numNodes,:] = new_top_nodes[j*numNodes:(j+1)*numNodes,:]
        save_adj[i*numNodes:(i+1)*numNodes,:] = new_top_adj[j*numNodes:(j+1)*numNodes,:]
    np.savetxt('../../dataSet/'+str(numIter)+'/nodes.csv', save_nodes, delimiter=",")
    np.savetxt('../../dataSet/'+str(numIter)+'/perturbation.csv', save_ptb, delimiter=",")
    sparse.save_npz('../../dataSet/'+str(numIter)+'/adj.npz', sparse.csr_matrix(save_adj))

    # sparse.save_npz('../../dataSet/'+str(numIter)+'/adj_conn.npz', sparse.csr_matrix(adj_conn_list))
    # np.savetxt('../../dataSet/'+str(numIter)+'/label.csv', np.array(z_np), delimiter = ",")