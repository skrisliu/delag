# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 19:49:16 2024

@author: skrisliu

Full data training
ATC only

nyc

#%% split. 1733,1667
# 0-400-1100-1733
# 0-750-1667

"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data 
from torch.utils.data import Dataset, DataLoader
import time
import argparse


site = 'nyc'
year = '2023'

## number of training samples per class
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--mode', type=int, default = 1)
args = parser.parse_args()
MODE = args.mode

#%% cuda
print(torch.__version__)
print('CUDA:',torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#%% Load data
# ERA5 in Kelvin
era5lst = np.load(site + '/datacube/era5lst_'+year +'.npy')

# Clear image close to zeros
fp = site + '/datacube/' + site + year + 'meanbands.npy'
ims = np.load(fp)
ims = np.transpose(ims,[2,0,1])
ims = ims/5000

# LST in Kelvin
fp = site + '/datacube/' + site + year + 'lsts.npy'
lsts = np.load(fp)
mask = lsts==0
lsts = lsts*0.00341802 + 149.0
lsts[mask] = 0

# cloud masks
fp = site + '/datacube/' + site + year + 'clearmasks.npy'
clearmasks = np.load(fp)
im1z, im1x, im1y = clearmasks.shape


# 0-570-1140-1733
# 0-560-1120-1667
# 0-400-1100-1733
# 0-750-1667
if MODE==1:
    x1,x2 = 0,570
    y1,y2 = 0,560
elif MODE==2:
    x1,x2 = 570,1140
    y1,y2 = 0,560
elif MODE==3:
    x1,x2 = 1140,1733
    y1,y2 = 0,560
elif MODE==4:
    x1,x2 = 0,570
    y1,y2 = 560,1120
elif MODE==5:
    x1,x2 = 570,1140
    y1,y2 = 560,1120
elif MODE==6:
    x1,x2 = 1140,1733
    y1,y2 = 560,1120
elif MODE==7:
    x1,x2 = 0,570
    y1,y2 = 1120,1667
elif MODE==8:
    x1,x2 = 570,1140
    y1,y2 = 1120,1667
elif MODE==9:
    x1,x2 = 1140,1733
    y1,y2 = 1120,1667

era5lst = era5lst[:,x1:x2,y1:y2]
ims = ims[:,x1:x2,y1:y2]
clearmasks = clearmasks[:,x1:x2,y1:y2]
lsts = lsts[:,x1:x2,y1:y2]

im1z, im1x, im1y = clearmasks.shape


#%% network
class atcnet(nn.Module):
    def __init__(self, imz=7):
        super(atcnet, self).__init__()
        
        self.matrix1 = nn.Parameter(torch.randn(im1x, im1y))
        self.matrix2 = nn.Parameter(torch.randn(im1x, im1y))
        self.matrix3 = nn.Parameter(torch.randn(im1x, im1y))
        self.matrix4 = nn.Parameter(torch.randn(im1x, im1y))

        a = np.arange(365)+1
        base = np.zeros([365,im1x,im1y],dtype=np.float32)
        for i in range(im1x):
            for j in range(im1y):
                base[:,i,j] = a
        self.base = torch.Tensor(base).to(device)

    # forward
    def forward(self, x):
        out = self.matrix1 + self.matrix2*10*torch.cos( 0.017214206320547945*(self.base-self.matrix3*100 ) ) + self.matrix4*era5lst  
        return out




#%% data to torch
era5lst = torch.from_numpy(era5lst)
ims = ims.reshape([1,7,im1x,im1y])
ims = torch.from_numpy(ims)
clearmasks = torch.from_numpy(clearmasks)
lsts = torch.from_numpy(lsts)

## to gpu
era5lst = era5lst.to(device)
ims = ims.to(device)
clearmasks = clearmasks.to(device)
lsts = lsts.to(device)


#%% build model
model = atcnet(imz=7)
model = model.to(device)
print(model)



#%% Training 1
criterion = torch.nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

log_loss = []
for i in range(300):
    outputs = model(ims)
    loss = criterion(outputs[clearmasks], lsts[clearmasks])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(i, loss.item())
    log_loss.append(loss.item())
    if i%50==5:
        time.sleep(4)
    
#%% Training 2
for j in range(2):
    criterion = torch.nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for i in range(250):
        outputs = model(ims)
        loss = criterion(outputs[clearmasks], lsts[clearmasks])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(i, loss.item())
        log_loss.append(loss.item())
        if i%50==5:
            time.sleep(5)


#%% Training 5
optimizer = optim.Adam(model.parameters(), lr=0.001)
count = 0
for i in range(1200):
    outputs = model(ims)
    loss = criterion(outputs[clearmasks], lsts[clearmasks])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(i, loss.item())
    log_loss.append(loss.item())
    if i%50==5:
        time.sleep(5)
    if i>=400:
        if i%4==0:
            count += 1
            impre = outputs.detach().cpu().numpy()
            imout = (impre-149.0)/0.00341802
            imout = np.uint16(imout)
            # time.sleep(0.5)
            for day in range(365):
                newpath = 'S:/tmp2/'+site+str(MODE)+'/save200/doy'+format(day+1,'03')
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                np.save(newpath+'/prea'+format(count,'03d')+'.npy', imout[day,:,:])
            













































