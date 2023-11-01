
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

from itertools import combinations_with_replacement
import pandas as pd

import scipy
torch.manual_seed(0)
# ***** parameters *****

faceNode = None;
mirror = 1
split = None
dim = 3
dist = 0.5
# num_dof = 51

nodesInit = np.genfromtxt('nodesInit.csv', delimiter=",")
numNodes = nodesInit.shape[0]

# First run `generation.py` script to obtain the base datset for sampling
sampleFolder = '20000'
adj_list =  sparse.load_npz('../../dataSet/'+sampleFolder+'/adj.npz').toarray()
numIter = int(adj_list.shape[0]/numNodes)
print(adj_list.shape[0])
L = np.arange(numIter)
all_combinations = [comb for comb in combinations_with_replacement(L, 2)]

tmp = []
for k in range(len(all_combinations)):
    idx1, idx2 = all_combinations[k][0], all_combinations[k][0]
    newA = adj_list[idx1*numNodes:(idx1+1)*numNodes,:] + adj_list[idx2*numNodes:(idx2+1)*numNodes,:]
    newA += newA.transpose()
    newA[newA != 0.]= 1.

    newA[np.triu_indices(numNodes)] = 0.
    row = np.nonzero(newA)[0]
    col = np.nonzero(newA)[1]
    g = nx.Graph()
    for i in range(row.shape[0]):
        g.add_edge(row[i], col[i])
    tmp.append(len(list(list(connected_components(g)))))
connected_part = np.array(tmp)
non_connected_id = np.where(connected_part != 1.)[0]
valid_comb = np.delete(all_combinations, non_connected_id, axis = 0)
print("Total number of unique combinations = ", len(all_combinations))
print("Number of valid combinations = ", len(valid_comb))

comb_idx = np.arange(len(valid_comb))
num_ptb = 20
num_unq_comb = 20000
comb_idx = np.random.choice(np.arange(len(valid_comb)), num_unq_comb, replace = False)
# comb_idx = genfromtxt('../../dataSet/90000/comb_idx.csv', delimiter = ",")
comb_idx = comb_idx.astype(int)

ptb_mode = 'continuously'
n_to_sample = int(num_ptb*num_unq_comb)
new_ptb = np.zeros([n_to_sample*numNodes, dim])
new_adj = np.zeros([n_to_sample*numNodes, numNodes])
new_nodes = np.zeros([n_to_sample*numNodes, dim])

# ptb_vec = np.zeros([num_dof])
upper = np.zeros([numNodes, dim])
lower = np.zeros([numNodes, dim])

x_edge_nodes = [1, 7, 19, 25]
y_edge_nodes = [3, 5, 21, 23]
z_edge_nodes = [9, 11, 15, 17]

xy_face_nodes = [4, 22]
yz_face_nodes = [12, 14]
xz_face_nodes = [10, 16]

body_nodes = [13]
    
dof_nodes = x_edge_nodes + y_edge_nodes + z_edge_nodes + xy_face_nodes + yz_face_nodes + xz_face_nodes + body_nodes

def get_dof_idx(node_number):
    i = node_number
    if i in x_edge_nodes:
        dof_idx = [3*i]

    elif i in y_edge_nodes:
        dof_idx = [3*i + 1]

    elif i in z_edge_nodes:
        dof_idx = [3*i + 2]

    elif i in xy_face_nodes:
        dof_idx = [3*i + 0, 3*i + 1]

    elif i in yz_face_nodes:
        dof_idx = [3*i + 1, 3*i + 2]

    elif i in xz_face_nodes:
        dof_idx = [3*i + 0, 3*i + 2]

    elif i in body_nodes:
        dof_idx = [3*i + 0, 3*i + 1, 3*i + 2]
    return dof_idx


def randomPtb(j, ptb_last_iter):
    ptb_vec = ptb_last_iter.flatten()
    dof_idx = get_dof_idx(j)
    ptb_tmp = randrange(len(dof_idx), -dist, dist)
    for j in range(len(dof_idx)):
        k = dof_idx[j]
        ptb_vec[k] += ptb_tmp[j]
    ptb = ptb_vec.reshape([numNodes, dim])
    return ptb

def ptbNodes(adj):
    non_zero_idx = np.where(np.sum(adj, axis = 0) != 0.)[0]
    ptb_iter = np.zeros([numNodes, dim])
    for j in non_zero_idx:
        if j in dof_nodes:
            to_move = np.random.choice([0,1],1)
            if to_move == 1:
                ptb_next_iter = randomPtb(j, ptb_iter)
                ptb_iter = ptb_next_iter.copy()
    return ptb_iter

def bary_coor1D(r):
    return r + 0.5

def bary_coor2D(s, t):
    m1, n1 = 1, 1
    m2, n2 = 1, 0
    m3, n3 = 0, 0
    m4, n4 = 0, 1

    N1 = 1./4*(1-s)*(1-t)
    N2 = 1./4*(1-s)*(1+t)
    N3 = 1./4*(1+s)*(1+t)
    N4 = 1./4*(1+s)*(1-t)
    ms = N1*m1 + N2*m2 + N3*m3 + N4*m4
    ns = N1*n1 + N2*n2 + N3*n3 + N4*n4
    return ms, ns

def bary_coor3D(s, t, q):
    x1, y1, z1 = 1, 1, 1
    x2, y2, z2 = 1, 1, 0
    x3, y3, z3 = 1, 0, 1
    x4, y4, z4 = 1, 0, 0
    x5, y5, z5 = 0, 1, 1
    x6, y6, z6 = 0, 1, 0
    x7, y7, z7 = 0, 0, 1
    x8, y8, z8 = 0, 0, 0

    N1 = 1./8*(1-s)*(1-t)*(1-q)
    N2 = 1./8*(1-s)*(1-t)*(1+q)
    N3 = 1./8*(1-s)*(1+t)*(1-q)
    N4 = 1./8*(1-s)*(1+t)*(1+q)
    N5 = 1./8*(1+s)*(1-t)*(1-q)
    N6 = 1./8*(1+s)*(1-t)*(1+q)
    N7 = 1./8*(1+s)*(1+t)*(1-q)
    N8 = 1./8*(1+s)*(1+t)*(1+q)
    
    xs = N1*x1 + N2*x2 + N3*x3 + N4*x4 + N5*x5 + N6*x6 + N7*x7 + N8*x8
    ys = N1*y1 + N2*y2 + N3*y3 + N4*y4 + N5*y5 + N6*y6 + N7*y7 + N8*y8
    zs = N1*z1 + N2*z2 + N3*z3 + N4*z4 + N5*z5 + N6*z6 + N7*z7 + N8*z8
    
    return xs, ys, zs

def ptb2nodes(new_ptb):

    nodes_n = nodesInit.copy()

    for n in x_edge_nodes:
        nodes_n[n,0] = bary_coor1D(new_ptb[n, 0]) 
    for n in y_edge_nodes:
        nodes_n[n,1] = bary_coor1D(new_ptb[n, 1]) 
    for n in z_edge_nodes:
        nodes_n[n,2] = bary_coor1D(new_ptb[n, 2])

    for n in xy_face_nodes:
        s, t = new_ptb[n,0], new_ptb[n,1]
        nodes_n[n,0], nodes_n[n,1] = bary_coor2D(s, t)

    for n in yz_face_nodes:
        s, t = new_ptb[n,1], new_ptb[n,2]
        nodes_n[n,1], nodes_n[n,2] = bary_coor2D(s, t)

    for n in xz_face_nodes:
        s, t = new_ptb[n,0], new_ptb[n,2]
        nodes_n[n,0], nodes_n[n,2] = bary_coor2D(s, t)

    for n in body_nodes:
        s, t, q= new_ptb[n,0], new_ptb[n,1], new_ptb[n,2]
        nodes_n[n,0], nodes_n[n,1], nodes_n[n,2] = bary_coor3D(s, t, q)
    return nodes_n

for k in range(len(comb_idx)):
    idx1, idx2 = valid_comb[comb_idx[k]][0], valid_comb[comb_idx[k]][1]
    newA = adj_list[idx1*numNodes:(idx1+1)*numNodes,:] + adj_list[idx2*numNodes:(idx2+1)*numNodes,:]
    newA += newA.transpose()
    newA[newA != 0.]= 1.
    ptb_nodes = True
    for m in range(num_ptb):
        if ptb_nodes == True:
            ptb_tmp = ptbNodes(newA)
        else:
            ptb_tmp = np.zeros([numNodes, dim])
        new_adj[(k*num_ptb+m)*numNodes:(k*num_ptb+m+1)*numNodes,:] = newA.copy()
        new_ptb[(k*num_ptb+m)*numNodes:(k*num_ptb+m+1)*numNodes,:] = ptb_tmp.copy()

        if np.all(ptb_tmp == 0 ):
            new_nodes[(k*num_ptb+m)*numNodes:(k*num_ptb+m+1)*numNodes,:] = nodesInit.copy()
        else:
            new_nodes[(k*num_ptb+m)*numNodes:(k*num_ptb+m+1)*numNodes,:] = ptb2nodes(ptb_tmp)
                

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

tmp = []
new_base_lattice = np.array(new_adj)
numUC = int(new_base_lattice.shape[0]/numNodes)
print("Total number of samples = ", numUC)
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
print("Number of non-connected uc = ", len(non_connected_id))
invalid_row = []
for j in non_connected_id:
    invalid_row_tmp = np.arange(j*numNodes, (j+1)*numNodes)
    invalid_row.extend(invalid_row_tmp)
invalid_row = np.array(invalid_row)
if invalid_row.shape[0] > 0:
    valid_new_adj = np.delete(new_base_lattice, invalid_row, axis = 0)
    valid_new_nodes = np.delete(new_nodes, invalid_row, axis = 0)
    valid_new_ptb = np.delete(new_ptb, invalid_row, axis = 0)
else:
    valid_new_adj = new_base_lattice.copy()
    valid_new_nodes = new_nodes.copy()
    valid_new_ptb = new_ptb.copy()

numUC = int(valid_new_adj.shape[0]/numNodes)
not_on_boundary = check_bc_connected(valid_new_adj, valid_new_nodes, numUC)
print("1st iter: number of unit cells = ", numUC)

file_invalid_id = not_on_boundary
if file_invalid_id.shape[0] > 0:
    invalid_row = []
    for j in file_invalid_id:
        invalid_row_tmp = np.arange(j*numNodes, (j+1)*numNodes)
        invalid_row.extend(invalid_row_tmp)
    invalid_row = np.array(invalid_row)
    new_top_adj = np.delete(valid_new_adj, invalid_row, axis = 0)
    new_top_nodes = np.delete(valid_new_nodes, invalid_row, axis = 0)
    new_top_ptb = np.delete(valid_new_ptb, invalid_row, axis = 0)
else:
    new_top_adj = valid_new_adj.copy()
    new_top_nodes = valid_new_nodes.copy()
    new_top_ptb = valid_new_ptb.copy()

numUC = int(new_top_adj.shape[0]/numNodes)
not_on_boundary = check_bc_connected(new_top_adj, new_top_nodes, numUC)

if not_on_boundary.shape[0] > 0:
    print("There are invalid unit cells.")
else:
    print("number of unit cells = ", numUC)
    # ptb = np.zeros(new_top_nodes.shape)
    # for i in range(numUC):
    #     tmp = new_top_ptb[i*numNodes:(i+1)*numNodes,:] 

    #     adj_tmp = new_top_adj[i*numNodes:(i+1)*numNodes,:]
    #     adj_tmp += adj_tmp.transpose()
    #     valence = np.sum(adj_tmp, axis = 0)
    #     zero_row = np.where(valence == 0.)[0]

    #     tmp[zero_row, :] = 0.
    #     ptb[i*numNodes:(i+1)*numNodes,:] = tmp
    #     new_top_nodes[i*numNodes:(i+1)*numNodes,:] = ptb2nodes(tmp)
    np.savetxt('../../dataSet/'+str(n_to_sample)+'/nodes.csv', new_top_nodes, delimiter=",")
    np.savetxt('../../dataSet/'+str(n_to_sample)+'/perturbation.csv', new_top_ptb, delimiter=",")
    sparse.save_npz('../../dataSet/'+str(n_to_sample)+'/adj.npz', sparse.csr_matrix(new_top_adj))
    # sparse.save_npz('../../dataSet/'+str(n_to_sample)+'/adj_conn.npz', sparse.csr_matrix(adj_conn_list))
    # np.savetxt('../../dataSet/'+str(numIter)+'/label.csv', np.array(z_np), delimiter = ",")