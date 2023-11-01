
import numpy as np
from numpy import genfromtxt
from itertools import combinations as comb

def T(c):
    a = np.array(list(c))
    return a

def str2stru(x):
    if x == 'A1':
        a = 1
        stack = 'sc'
        s = ''
    elif x == 'A2':
        a = 2
        stack = 'sc'
        s = '_S'
    elif x == 'B2':
        a = 0
        stack = 'bcc'
        s = '_S'
    elif x == 'C2':
        a = 3
        stack = 'fcc'
        s = '_S'
    return a, stack, s


def randrange(n, vmin, vmax):
    """
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    """
    return (vmax - vmin)*np.random.rand(n) + vmin


def moveNodes(model, stack, fileS, s, f, nodes):

    baseTop = None
    dist = 0.25
    move_random = 1
    if move_random == 1:
        move = np.random.choice([1,0],1)
    else:
        move = 1
    nodesInit = np.genfromtxt('nodesInit.csv', delimiter=",")
    numNodes = nodes.shape[0]
    
    if model == 'sc':     
        if fileS == '_S':

            corner = np.array([8, 28, 32, 34])
            disp_tmp = np.zeros([4,3])
            for i in range(4):
                disp_tmp[i,:] = randrange(3, -dist, 0)
            disp_tmp[0, :] = randrange(3, -dist, 0); disp_tmp[0, 2] = 0.
            disp_tmp[1, :] = randrange(3, -dist, 0); disp_tmp[1, 1] = 0.
            disp_tmp[2, :] = randrange(3, -dist, 0); disp_tmp[2, 0] = 0.
            disp_tmp[3, :] = 0.

            for count in range(corner.shape[0]):
                rowId = corner[count]
                # move = np.random.choice([1,0,1],1) 
                if move_random == 1:
                    move = np.random.choice([1,0],1)
                if move == 1:
                    nodes[rowId,:] += disp_tmp[count, :]

            to_move_id = np.array([21, 29, 33])
            disp_tmp = np.zeros([3, 3])
            disp_tmp[0,:] = randrange(3, -dist, 0); disp_tmp[0, 2] = randrange(1, -dist, dist);
            disp_tmp[1,:] = randrange(3, -dist, 0); disp_tmp[1, 1] = randrange(1, -dist, dist);
            disp_tmp[2,:] = randrange(3, -dist, 0); disp_tmp[2, 0] = randrange(1, -dist, dist);

            for k in range(len(to_move_id)):
                disp = disp_tmp[k,:]
                mid_id = to_move_id[k]
                # move = np.random.choice([1,0,1],1) 
                if move_random == 1:
                    move = np.random.choice([1,0],1)
                if move == 1:
                    nodes[mid_id,:] += disp

            # idx = []
            # args = (nodesInit[:,0]==1.)&(nodesInit[:,1]==1.)&(nodesInit[:,2]==1.);idx.append(np.where(args)[0])
            # col = np.array(idx)[0]

            # idx = []
            # args = (nodesInit[:,0]==1.)&(nodesInit[:,1]==0.)&(nodesInit[:,2]==1.);id_1 = np.where(args)[0]; idx.append(id_1)  # 1
            # if move_random == 1:
            #     move = np.random.choice([1,0],1)
            # if move == 1:
            #     if baseTop == True:
            #         disp = np.random.choice([1,-1,0],3) * dist
            #     else:
            #         disp = randrange(3, -dist, 0)
            #         # disp = randrange(3, -dist, dist)
            #     disp[1] = 0.
            #     nodes[id_1, :] += disp


            #     disp = randrange(3, -dist, 0)
            #     disp[1] = randrange(1, -dist, dist)
            #     midNode = (nodesInit[id_1,:] + nodesInit[col[0], :])*1./2
            #     midNode = midNode[0]
            #     args = (nodesInit[:,0]==midNode[0])&(nodesInit[:,1]==midNode[1])&(nodesInit[:,2]==midNode[2]);mid_id = np.where(args)[0]
            #     nodes[mid_id,:] += disp

            # args = (nodesInit[:,0]==0)&(nodesInit[:,1]==1.)&(nodesInit[:,2]==1.);id_2 = np.where(args)[0]; idx.append(id_2)  # 2
            # if move_random == 1:
            #     move = np.random.choice([1,0],1)
            # if move == 1:
            #     if baseTop == True:
            #         disp = np.random.choice([1,-1,0],3) * dist
            #     else:
            #         disp = randrange(3, -dist, 0)
            #         # disp = randrange(3, -dist, dist)
            #     disp[0] = 0.
            #     nodes[id_2,:] += disp

            #     disp = randrange(3, -dist, 0)
            #     disp[0] = randrange(1, -dist, dist)
            #     midNode = (nodesInit[id_2,:] + nodesInit[col[0], :])*1./2
            #     midNode = midNode[0]
            #     args = (nodesInit[:,0]==midNode[0])&(nodesInit[:,1]==midNode[1])&(nodesInit[:,2]==midNode[2]);mid_id = np.where(args)[0]
            #     nodes[mid_id,:] += disp

            # args = (nodesInit[:,0]==1.)&(nodesInit[:,1]==1.)&(nodesInit[:,2]==0.);id_3 = np.where(args)[0]; idx.append(id_3) # 3

            # if move_random == 1:
            #     move = np.random.choice([1,0],1)
            # if move == 1:
            #     if baseTop == True:
            #         disp = np.random.choice([1,-1,0],3) * dist
            #     else:
            #         disp = randrange(3, -dist, 0)
            #     disp[2] = 0.
            #     nodes[id_3,:] += disp

            #     disp = randrange(3, -dist, 0)
            #     disp[2] = randrange(1, -dist, dist)
            #     midNode = (nodesInit[id_3,:] + nodesInit[col[0], :])*1./2
            #     midNode = midNode[0]
            #     args = (nodesInit[:,0]==midNode[0])&(nodesInit[:,1]==midNode[1])&(nodesInit[:,2]==midNode[2]);mid_id = np.where(args)[0]
            #     nodes[mid_id,:] += disp
            # 
            # row = np.array(idx)
            # idx = []
            # args = (nodesInit[:,0]==1.)&(nodesInit[:,1]==1.)&(nodesInit[:,2]==1.);idx.append(np.where(args)[0])
            # col = np.array(idx)[0]
            
            # # A_scS = np.zeros([numNodes,numNodes])
            # for i in row:
            #     midNode = (nodesInit[i[0],:] + nodesInit[col[0], :])*1./2
            #     args = (nodesInit[:,0]==midNode[0])&(nodesInit[:,1]==midNode[1])&(nodesInit[:,2]==midNode[2]);mid_id = np.where(args)[0]
            #     if move == 1:
            #         disp = randrange(3, -dist, 0)


            #     A_scS[i,mid_id] = 1
            #     A_scS[mid_id,i] = 1
                
            #     A_scS[col,mid_id] = 1
            #     A_scS[mid_id,col] = 1
            
            # np.savetxt('baseLattices/' + model + '/A_S.csv', A_scS, delimiter=",")
        elif fileS == '':
            A_sc = np.zeros([numNodes, numNodes])
            for count in [1,2,3]:
                mididx = []
                if count == 1:
                    to_move_id = np.array([29, 31, 3, 5]); axisFixed = 0; const_axis = 1
                    # edges = np.array([[28,34], [32,26], [2,8], [0,6]]) 
                elif count == 2:
                    to_move_id =np.array([33, 27, 7, 1]); axisFixed = 1; const_axis = 0
                    # edges = np.array([[34,32], [26,28], [8,6], [0,2]]) ; axisFixed = 1; const_axis = 0

                elif count == 3:
                    to_move_id = np.array([21, 19, 15, 13]); axisFixed = 0; const_axis = 2
                    # edges = np.array([[8,34], [6,32], [2,28], [26,0]]) ; axisFixed = 0; const_axis = 2      
                disp_tmp = np.zeros([4,3])
                disp_tmp[0,:] = randrange(3, -dist, 0); disp_tmp[0, const_axis] = randrange(1, -dist, dist);
                disp_tmp[1,:] = randrange(3, -dist, 0); disp_tmp[1, const_axis] = randrange(1, -dist, dist); disp_tmp[1, axisFixed] = randrange(1, 0, dist);
                disp_tmp[2,:] = randrange(3, 0, dist); disp_tmp[2, const_axis] = randrange(1, -dist, dist); disp_tmp[2,axisFixed] = randrange(1, -dist, 0);
                disp_tmp[3,:] = randrange(3, 0, dist); disp_tmp[3, const_axis] = randrange(1, -dist, dist);

                for k in range(len(to_move_id)):
                    disp = disp_tmp[k,:]
                    mid_id = to_move_id[k]
                    # move = np.random.choice([1,0,1],1) 
                    if move_random == 1:
                        move = np.random.choice([1,0],1)
                    if move == 1:
                        nodes[mid_id,:] += disp
                    # mididx.append(mid_id)
                # mididx = np.array(mididx)

            corner = np.array([0, 2, 6, 8, 26, 28, 32, 34])
            disp_tmp = np.zeros([8,3])
            disp_tmp[0, :] = 0.
            disp_tmp[1, 0] = randrange(1, -dist, 0)
            disp_tmp[2, 1] = randrange(1, -dist, 0)
            disp_tmp[3, :] = randrange(3, -dist, 0); disp_tmp[3, 2] = 0.
            disp_tmp[4, 2] = randrange(1, -dist, 0)
            disp_tmp[5, :] = randrange(3, -dist, 0); disp_tmp[5, 1] = 0.
            disp_tmp[6, :] = randrange(3, -dist, 0); disp_tmp[6, 0] = 0.
            disp_tmp[7, :] = 0.

            for count in range(corner.shape[0]):
                rowId = corner[count]
                # move = np.random.choice([1,0,1],1) 
                if move_random == 1:
                    move = np.random.choice([1,0],1)
                if move == 1:
                    nodes[rowId,:] += disp_tmp[count, :]

                
                # for i in range(mididx.shape[0]):
            
                #     A_sc[mididx[i],edges[i]-1] = 1
                #     A_sc[edges[i]-1,mididx[i]] = 1
            # np.savetxt('baseLattices/' + model + '/nodes.csv', nodes, delimiter=",")
            # np.savetxt('baseLattices/' + model + '/A.csv', A_sc, delimiter=",")
    elif model == 'bcc':
        if fileS == '_S':            
            split = None
            # args = (nodesInit[:,0]==1.)&(nodesInit[:,1]==1.)&(nodesInit[:,2]==1.);col = np.where(args)[0]
            # if move == 1:
                # nodes[idx,:] += randrange(3, -dist, 0)
            # splitNode = []
            # if split == 1:
            #     args = (nodesInit[:,0]==1./2)&(nodesInit[:,1]==0.)&(nodesInit[:,2]==1./2);splitNode.append(np.where(args)[0])
            #     args = (nodesInit[:,0]==1./2)&(nodesInit[:,1]==1./2)&(nodesInit[:,2]==0.);splitNode.append(np.where(args)[0])
            #     args = (nodesInit[:,0]==0)&(nodesInit[:,1]==1./2)&(nodesInit[:,2]==1./2);splitNode.append(np.where(args)[0])
            # #     nodes[splitNode,:] = nodes[splitNode,:] + disp
            #     row = np.concatenate((np.array(idx),np.array(splitNode)))
            # else:
                # args = (nodesInit[:,0]==0)&(nodesInit[:,1]==0.)&(nodesInit[:,2]==0.);
                # idx0 = np.where(args)[0]
            if baseTop == True:
                disp = np.random.choice([1,-1,0],3) * dist
            else:
                disp = randrange(3, -dist, dist)

            to_move_id = np.array([9, 17, 24])
            # args = (nodesInit[:,0]==1./2)&(nodesInit[:,1]==1./2)&(nodesInit[:,2]==1./2);idx1 = np.where(args)[0]
            # move = np.random.choice([1,0,1],1) 
            for idx in to_move_id:
                if move_random == 1:
                    move = np.random.choice([1,0],1)
                # if split != 1:
                if move == 1:
                    nodes[to_move_id,:] += disp
            # corner = np.array([0, 34])
            # disp_tmp = np.zeros([2,3])
            # disp_tmp[0, :] = 0.
            # disp_tmp[1, :] = randrange(3, -dist, 0);
            # for count in range(corner.shape[0]):
            #     rowId = corner[count]
            #     # move = np.random.choice([1,0,1],1) 
            #     if move_random == 1:
            #         move = np.random.choice([1,0],1)
            #     if move == 1:
            #         nodes[rowId,:] += disp_tmp[count, :]
            # args = (nodesInit[:,0]==1./4)&(nodesInit[:,1]==1./4)&(nodesInit[:,2]==1./4);idx2 = np.where(args)[0]
            # args = (nodesInit[:,0]==3./4)&(nodesInit[:,1]==3./4)&(nodesInit[:,2]==3./4);idx3 = np.where(args)[0]
            
            # col = np.array(idx)
            
            # A_bccS = np.zeros([numNodes,numNodes])
            # # # for i in row:
            # A_bccS[idx0,idx2] = 1; A_bccS[idx2,idx0] = 1
            # A_bccS[idx1,idx2] = 1; A_bccS[idx2,idx1] = 1
            # A_bccS[idx1,idx3] = 1; A_bccS[idx3,idx1] = 1
            # A_bccS[col,idx3] = 1; A_bccS[idx3, col] = 1
            
            # row = np.nonzero(A_bccS)[0]
            # col = np.nonzero(A_bccS)[1]

            # np.savetxt('baseLattices/' + model + '/A_S.csv', A_bccS, delimiter=",")
        else:
            split = None
            newNodes = None
            distRange = 0.2
            # octant
            # row = np.array([17])
            # col = np.array([8,6,0,2,34,32,26,28])
            # if baseTop == True:
            #     disp = np.random.choice([1,-1,0],3) * dist
            # else:
            #     disp = randrange(3, -dist/2, dist/2)
            # nodes = nodes.copy()
            to_move_id = np.array([9, 10, 11, 12, 22, 23, 24, 25, 17])
            for idx in to_move_id:
                disp = randrange(3, -distRange, distRange)
                if move_random == 1:
                    move = np.random.choice([1,0],1)
                if move == 1:
                    nodes[idx,:] += disp

            # for i in row:
            #     for j in col:
            #         disp = randrange(3, -distRange, distRange)
            #         idx = []
            #         mid = np.zeros([1,3])
            #         mid[:,:] = (nodesInit[i, :] + nodesInit[j, :])*1./2
            #         if newNodes == 1:
            #             args = ((nodes, mid))
            #             nodes = np.concatenate(args, axis = 0)
            #         args = (nodesInit[:,0]==mid[:,0])&(nodesInit[:,1]==mid[:,1])&(nodesInit[:,2]==mid[:,2]);idx.append(np.where(args)[0])
            #         # move = np.random.choice([1,0,1],1) 
            #         if move_random == 1:
            #             move = np.random.choice([1,0],1)
            #         if move == 1:
            #             nodes[idx,:] += disp
            # disp = randrange(3, -distRange, distRange)
            # args = (nodesInit[:,0]==1./2)&(nodesInit[:,1]==1./2)&(nodesInit[:,2]==1./2);idxR = np.where(args)[0]
            # # move = np.random.choice([1,0,1],1) 
            # if move_random == 1:
            #     move = np.random.choice([1,0],1)
            # if move == 1:
            #     nodes[idxR,:] += disp
            corner = np.array([0, 2, 6, 8, 26, 28, 32, 34])
            disp_tmp = np.zeros([8,3])
            disp_tmp[0, :] = 0.
            disp_tmp[1, 0] = randrange(1, -dist, 0)
            disp_tmp[2, 1] = randrange(1, -dist, 0)
            disp_tmp[3, :] = randrange(3, -dist, 0); disp_tmp[3, 2] = 0.
            disp_tmp[4, 2] = randrange(1, -dist, 0)
            disp_tmp[5, :] = randrange(3, -dist, 0); disp_tmp[5, 1] = 0.
            disp_tmp[6, :] = randrange(3, -dist, 0); disp_tmp[6, 0] = 0.
            disp_tmp[7, :] = 0.

            for count in range(corner.shape[0]):
                rowId = corner[count]
                # move = np.random.choice([1,0,1],1) 
                if move_random == 1:
                    move = np.random.choice([1,0],1)
                if move == 1:
                    nodes[rowId,:] += disp_tmp[count, :]

            # corner = np.array([8,6,0,2,34,32,26,28])
            # disp_tmp = np.zeros([8,3])
            # for i in range(8):
            #     disp_tmp[i,:] = randrange(3, -dist, 0)
            # disp_tmp[0, 2] = 0.;
            # disp_tmp[1, 0] = 0.; disp_tmp[1, 2] = 0.;
            # disp_tmp[2, :] = 0.;
            # disp_tmp[3, 1] = 0; disp_tmp[3, 2] = 0.;
            # disp_tmp[4, :] = 0; 
            # disp_tmp[5, 0] = 0.;
            # disp_tmp[6, 0] = 0; disp_tmp[6, 1] = 0.;
            # disp_tmp[7, 1] = 0; 

            # for count in range(corner.shape[0]):
            #     rowId = corner[count]
            #     # move = np.random.choice([1,0,1],1) 
            #     if move_random == 1:
            #         move = np.random.choice([1,0],1)
            #     if move == 1:
            #         nodes[rowId,:] += disp_tmp[count, :]
             
            # disp = randrange(3, -dist/2, dist/2)
            # nodes[row - 1,:] += disp
            # np.savetxt('baseLattices/' + model + '/nodes.csv', newCoor, delimiter=",")

            # A_bcc = np.zeros([numNodes,numNodes])
            # midIdx = np.arange(8) + 1 + 27
            # for k in range(midIdx.shape[0]):
            #     A_bcc[midIdx[k]-1, row-1] = 1
            #     A_bcc[row-1, midIdx[k]-1] = 1
            
            #     A_bcc[midIdx[k]-1, col[k]-1] = 1
            #     A_bcc[col[k]-1, midIdx[k]-1] = 1
            # np.savetxt('baseLattices/' + model + '/A.csv', A_bcc, delimiter=",") 
    elif model == 'fcc':
        split = None
        if fileS == '_S':
            if baseTop == True:
                disp = np.random.choice([1,-1,0],3) * dist
            # else:
            #     disp = randrange(3, -dist, dist)
            # A_fccS = np.zeros([numNodes, numNodes])
            to_move_id = np.array([16, 20, 30, 18, 14, 4])
            disp_tmp = np.zeros([6, 3])
            for i in range(disp_tmp.shape[0]):
                disp_tmp[i,:] = randrange(3, -dist, dist)
            disp_tmp[0, 0] = randrange(1, -dist, 0)
            disp_tmp[1, 1] = randrange(1, -dist, 0)
            disp_tmp[2, 2] = randrange(1, -dist, 0)
            disp_tmp[3, 0] = randrange(1, 0, dist)
            disp_tmp[4, 1] = randrange(1, 0, dist)
            disp_tmp[5, 2] = randrange(1, 0, dist)


            for k in range(len(to_move_id)):
                disp = disp_tmp[k,:]
                mid_id = to_move_id[k] 
                if move_random == 1:
                    move = np.random.choice([1,0],1)
                if move == 1:
                    nodes[mid_id,:] += disp

            # for count in [0, 1, 2, 3, 4, 5]:
            #     mididx = []
            #     if count == 0:
            #         midEdge = np.array([[26, 34]]);
            #         disp[2] = randrange(1, -dist, 0)
            #     elif count == 1:
            #         midEdge = np.array([[26,6]]); 
            #         disp[0] = randrange(1, 0, dist)
            #     elif count == 2:
            #         midEdge = np.array([[26,2]]); 
            #         disp[1] = randrange(1, 0, dist)
            #         # zero_idx = 0;disp[zero_idx] = 0.
            #     elif count == 3:
            #         midEdge = np.array([[6,2]]);
            #         disp[2] = randrange(1, 0, dist) 
            #         # zero_idx = 2;disp[zero_idx] = 0.
            #     elif count == 4:
            #         midEdge = np.array([[34, 6]]);
            #         disp[1] = randrange(1, -dist, 0)
            #     elif count == 5:
            #         midEdge = np.array([[34, 2]]);
            #         disp[0] = randrange(1, -dist, 0)


            #     for i in midEdge:
            #         midNode = sum(nodesInit[i,:])*1./2
            #         args = (nodesInit[:,0]==midNode[0])&(nodesInit[:,1]==midNode[1])&(nodesInit[:,2]==midNode[2]);mid_id = np.where(args)[0]
            #         # move = np.random.choice([1,0,1],1) 
            #         if move_random == 1:
            #             move = np.random.choice([1,0],1)
            #         if move == 1:
            #             nodes[mid_id,:] += disp
            #         mididx.append(mid_id)
            #     mididx = np.array(mididx)

            corner = np.array([2, 6, 26, 34])
            disp_tmp = np.zeros([4,3])
            disp_tmp[0, 0] = randrange(1, -dist, 0)
            disp_tmp[1, 1] = randrange(1, -dist, 0)
            disp_tmp[2, 2] = randrange(1, -dist, 0)
            for count in range(corner.shape[0]):
                rowId = corner[count]
                # move = np.random.choice([1,0,1],1) 
                if move_random == 1:
                    move = np.random.choice([1,0],1)
                if move == 1:
                    nodes[rowId,:] += disp_tmp[count, :]

                # for i in range(mididx.shape[0]):

                #     A_fccS[mididx[i],midEdge[i]-1] = 1
                #     A_fccS[midEdge[i]-1,mididx[i]] = 1
            # np.savetxt('baseLattices/' + model + '/A_S.csv', A_fccS, delimiter=",")
        else:
            if baseTop == True:
                disp = np.random.choice([1,-1,0],3) * dist
            else:
                disp = randrange(3, -dist, dist)
            
            newNodes = None
            # octant
            tetra = np.array([[14, 17, 22],[9, 14, 17],[17, 24, 9],[17, 24, 22],[14, 20, 9],[14, 20, 22],[20, 24, 9],[20, 24, 22]])
            tetraVertex = np.array([5, 1, 2, 6, 4, 8, 3, 7])
            A_fcc = np.zeros([numNodes, numNodes])
            count = 0
            for i,j in zip(tetraVertex,tetra):
                i = i.tolist()
                j = j.tolist()
                j.append(i)
                choose  = T(comb(j,2))
                for m in choose:
                    mid = np.zeros([1,3])
                    mid[:,:] = (nodesInit[m[0]-1, :] + nodesInit[m[1]-1, :])*1./2
                    idx = (nodesInit[:,0]==mid[:,0])&(nodesInit[:,1]==mid[:,1])&(nodesInit[:,2]==mid[:,2])
                    if idx.any() == False:
                        args = (nodes,mid)
                        if newNodes == 1:
                            nodes = np.concatenate(args, axis = 0)
                    else:
                        tempID = np.where(idx)[0]
                        if move == 1:
                            nodes[tempID,:] += disp
                        A_fcc[tempID, m-1] = 1
                        A_fcc[m-1, tempID] = 1
            # np.savetxt('baseLattices/' + model + '/nodes.csv', nodes, delimiter=",")

    return nodes


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

