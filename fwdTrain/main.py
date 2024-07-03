import torch
import numpy as np
from numpy import genfromtxt
from scipy import sparse
import pandas as pd
import time
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.autograd import Variable
from torch.utils.data import random_split

import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from models.parameters import *
from models.model import *
from models.utils import *
import torch.autograd.profiler as profiler
from tqdm import trange
from torch.optim.lr_scheduler import StepLR


torch.cuda.empty_cache()
torch.manual_seed(0)

def test(model, c_model, test_loader, saveResults, test_batch_size):

    model.eval()
    c_model.eval()

    adj_test_mse = 0.
    x_test_mse = 0.
    c_test_mse = 0.
    test_kld_loss = 0.

    x_test = []; adj_test = []
    x_pred = []; adj_pred = []

    with torch.no_grad():
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

            adj_mse, x_mse = recon_criterion(adj_decoded, adj), recon_criterion(x_decoded, x)
            adj_test_mse += adj_mse.item()
            x_test_mse += x_mse.item()

            test_kld = kld_loss(mu, std)
            test_kld_loss += test_kld.item()

            c_mse = stiffness_weighted_loss(c_pred, c)
            c_test_mse += c_mse.item()


    if saveResults == True:

        x_test = x.cpu().detach().numpy()
        x_pred = x_decoded.cpu().detach().numpy()

        adj_test = adj.cpu().detach().numpy()
        adj_pred = adj_decoded.cpu().detach().numpy()
        
        c_pred = c_pred.cpu().detach().numpy()
        c_test = c.cpu().detach().numpy()

        z_pred = encoded.cpu().detach().numpy()
        mu_pred = mu.cpu().detach().numpy()
        
        np.savetxt( outputFolder+'/x_test.csv', x_test, delimiter=",")
        np.savetxt( outputFolder+'/x_pred.csv', x_pred, delimiter=",")
        np.savetxt( outputFolder+'/adj_test.csv', adj_test, delimiter=",")
        np.savetxt( outputFolder+'/adj_pred.csv', adj_pred, delimiter=",")
        np.savetxt( outputFolder+'/c_test.csv', c_test, delimiter=",")
        np.savetxt( outputFolder+'/c_pred.csv', c_pred, delimiter=",")
        np.savetxt( outputFolder+'/z_pred.csv', z_pred, delimiter=",")
        np.savetxt( outputFolder+'/mu_pred.csv', mu_pred, delimiter=",")
        
    return adj_test_mse, x_test_mse, c_test_mse, test_kld

def train(epoch, KLweight):

    c_model.train()

    adj_loss_mse = 0.
    x_loss_mse = 0.
    c_loss_mse = 0.
    z_sim_loss = 0.

    data = train_prefetcher.next()
    n_batch = 0

    while data is not None:

        n_batch += 1
        if n_batch >= num_iters:
            break
        data = train_prefetcher.next()

        adj = data[0]; adj = adj.to(device)
        x = data[1]; x = x.to(device)
        c = data[2]; c = c.to(device)

        if load_pretrained_model == False:
            model.train()
            optimizer.zero_grad(set_to_none=True)
        else:
            model.eval()
            optimizer.zero_grad(set_to_none=True)

        encoded, mu, std = model.encoder(adj, x)
        if add_noise == True:
            c_input = encoded
        else:
            c_input = mu
            
        c_pred = c_model(c_input)

        adj_decoded, x_decoded = model.decoder(encoded)

        val_pred = torch.sum(adj_decoded, dim = 1)
        val = torch.sum(adj, dim = 1)

        adj_train_mse, x_train_mse = recon_criterion(adj_decoded, adj), recon_criterion(x_decoded, x)
        train_kld = kld_loss(mu, std)
        
        c_train_mse = stiffness_weighted_loss(c_pred, c)
        val_loss = recon_criterion(val, val_pred)

        if load_pretrained_model == False:
            loss = (adj_train_mse + val_loss)*a_weight + (train_kld) + x_train_mse*x_weight + c_train_mse*c_weight[epoch]
        else:
            loss = c_train_mse

        loss.backward()
        optimizer.step()
        adj_loss_mse += adj_train_mse.item()
        x_loss_mse += x_train_mse.item()
        c_loss_mse += c_train_mse.item()

    return adj_loss_mse, x_loss_mse, c_loss_mse, z_sim_loss

x_weight = 1e2
a_weight = 1
c_weight = np.ones([epochs])*1e4 # constant weight of stiffness prediction loss

model = vaeModel()
c_model = cModel()
model.to(device)
c_model.to(device)

torch.autograd.set_detect_anomaly(True)

if load_pretrained_model == True:
    model.load_state_dict(torch.load(savedModelFolder+'/best_model.pt'))
else:
    model.apply(weights_init)
c_model.apply(weights_init)

model.eval()
saveResults = None

dataset  = TensorDataset(adj_list, x, c_data)

num_train = len(dataset)
split = int(np.floor(valid_size * num_train))               # 5% of the data is used for validation
train_dataset, test_dataset = random_split(dataset = dataset, lengths = [num_train - split,split])
train_loader = DataLoaderX(train_dataset, batch_size = batch_size, shuffle = True, pin_memory = True)
test_loader = DataLoaderX(test_dataset, batch_size = test_batch_size, shuffle = True, pin_memory = True)
num_iters = len(train_loader)

if load_pretrained_model == True:
    optimizer = torch.optim.Adam(c_model.parameters(), lr = c_lr)
else:
    optimizer = torch.optim.Adam(list(model.parameters())+list(c_model.parameters()), lr = learningRate)
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

best_c_accuracy = 0.
best_adj_x_accuracy = 0.
best_accuracy_file = None
loss_file = outputFolder + '/loss.txt'
with open(loss_file, 'w') as the_file:
    the_file.write('EPOCH, Adj Train Loss, Adj Test Loss, X Train Loss, X Test Loss, C Train Loss, C Test Loss, Lr' +'\n')
the_file.close()

if best_accuracy_file is None:
    epoch_loss = [ [0 for col in range(8)] for row in range(epochs)]
    for epoch in trange(epochs):
        
        start1 = time.time()

        train_prefetcher = data_prefetcher(train_loader)
        KLweight = 1.0

        adj_loss_mse, x_loss_mse, c_loss_mse, z_sim_loss = train(epoch, KLweight)
        
        end1 = time.time()
        scheduler.step()
        
        if epoch == epochs - 1:
            saveResults = True

        adj_test_mse, x_test_mse, c_test_mse, test_kld = test(model, c_model, test_loader, saveResults, test_batch_size)
        
        epoch_loss[epoch][0] = epoch
        epoch_loss[epoch][1] = adj_test_mse
        epoch_loss[epoch][2] = x_test_mse
        epoch_loss[epoch][3] = c_test_mse

        epoch_loss[epoch][4] = adj_loss_mse
        epoch_loss[epoch][5] = x_loss_mse
        epoch_loss[epoch][6] = c_loss_mse
        epoch_loss[epoch][7] = optimizer.param_groups[0]['lr']

        if (epoch % 10 == 0.) or (epoch == epochs - 1):

            print("Epoch:", '%03d' % (epoch + 1),'/', str(epochs)\
            ,", adj train mse =", "{:.4f}".format(adj_loss_mse / len(train_loader)/batch_size)\
            ,", adj test mse =", "{:.4f}".format(adj_test_mse/len(test_loader)/test_batch_size)\
            ,", x train =", "{:.4f}".format(x_loss_mse/ len(train_loader)/batch_size )\
            ,", x test =", "{:.4f}".format(x_test_mse/len(test_loader)/test_batch_size)\
            ,", c train =", "{:.4f}".format(c_loss_mse/ len(train_loader)/batch_size )\
            ,", c test =", "{:.4f}".format(c_test_mse/len(test_loader)/test_batch_size)\
            ,", KL weight =", "{:.4f}".format(KLweight)\
            ,", time =", "{:.2f}".format(end1-start1))

            if best_c_accuracy == 0. or (c_test_mse/len(test_loader)/test_batch_size < best_c_accuracy):
                print('updating best c accuracy: previous best = {:.4f} new best = {:.4f}'.format(best_c_accuracy,
                                                                                         c_test_mse/len(test_loader)/test_batch_size))
                best_c_accuracy = c_test_mse/len(test_loader)/test_batch_size
                torch.save(c_model.state_dict(), outputFolder+'/best_c_model.pt')
                torch.save(model.state_dict(), outputFolder+'/best_model.pt')
            
model.cpu()
c_model.cpu()

model.eval()
c_model.eval()

torch.save(model.state_dict(), outputFolder+'/model.pt')
torch.save(c_model.state_dict(), outputFolder+'/c_model.pt')

loss_record = pd.DataFrame(epoch_loss, columns = ['EPOCH' , 'Adj Train Loss', 'Adj Test Loss','X Train Loss',  'X Test Loss','C Train Loss', 'C Test Loss', 'Lr'])
loss_record.to_csv(loss_file, index=False, sep=',')


