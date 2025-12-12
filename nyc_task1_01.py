# -*- coding: utf-8 -*-
"""
@author: skrisliu.com
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
import copy
from osgeo import gdal
import pandas as pd

#%% Setup
site = 'nyc'
year = '2023'

MODE = 326     # ---- prefixed code to distinguish task
d,y = 155,2023 # ---- cloudmask
dpre = 91      # ---- predict date

SPECTRALMODE = 'SEN'   # ---- LDS FOR LANDSAT, SEN FOR SENTINEL2. Default is SEN. 

fp00 = './'     # ---- data path
SAVEPATH = './save/'                # ---- save path


#%% cuda
print(torch.__version__)
print('CUDA:',torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#%% Load data
# ERA5 in Kelvin
era5lst = np.load(site + '/datacube/era5lst_'+year +'.npy')

# spectral bands
if SPECTRALMODE=='LDS':
    fp = fp00 + site + '/datacube/' + site + year + 'meanbands.npy'
    ims = np.load(fp)
    ims = np.transpose(ims,[2,0,1])
    ims = ims/5000
elif SPECTRALMODE=='SEN':
    fp = fp00 + site + '/datacube/' + site + year + 'meanbands_sen.npy'
    ims = np.load(fp)
    ims = ims/5000

# LST in Kelvin
fp = fp00 + site + '/datacube/' + site + year + 'lsts.npy'   # ---- stored as uint16 with standard Landsat conversion to save space
lsts = np.load(fp)
mask = lsts==0
lsts = lsts*0.00341802 + 149.0
lsts[mask] = 0

# cloud masks
fp = fp00 + site + '/datacube/' + site + year + 'clearmasks.npy'
clearmasks = np.load(fp)
im1z, im1x, im1y = clearmasks.shape


#%% Subset data, for task #1 testing on a subset of the data. 
x1,x2 = 550,900
y1,y2 = 650,1150


#%% subset data
clearmasks = clearmasks[:,x1:x2,y1:y2]
ims = ims[:,x1:x2,y1:y2]
lsts = lsts[:,x1:x2,y1:y2]
era5lst = era5lst[:,x1:x2,y1:y2]

im1z, im1x, im1y = clearmasks.shape

idx1 = np.array(np.where(clearmasks[d]==False)).T   # ---- cloud patterns from d
idx2 = idx1
clearmasks[dpre,idx2[:,0],idx2[:,1]] = False        # ---- predict date, make sure on the predict date, it has the cloud patten the same from d
SAVENEWMASK = False
if SAVENEWMASK:
    newpath = site + '/train' + str(MODE)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    np.save(newpath + '/' + site + '_mask'+str(MODE)+'_idx' + str(d) + '.npy', clearmasks)


#%% Model
IMZ = ims.shape[0]

# model
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
ims = ims.reshape([1,IMZ,im1x,im1y])
ims = torch.from_numpy(ims)
clearmasks = torch.from_numpy(clearmasks)
lsts = torch.from_numpy(lsts)

# to gpu
era5lst = era5lst.to(device)
ims = ims.to(device)
clearmasks = clearmasks.to(device)
lsts = lsts.to(device)



#%% build model
model = atcnet(imz=IMZ)
model = model.to(device)
print(model)



#%% Training
criterion = torch.nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

log_loss = []
for i in range(400):
    outputs = model(ims)
    loss = criterion(outputs[clearmasks], lsts[clearmasks])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(i, loss.item())
    log_loss.append(loss.item())

#%% SAVE ENSEMBLE
optimizer = optim.Adam(model.parameters(), lr=0.1)
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
            for day in range(365):
                newpath = SAVEPATH+site+str(MODE)+'/save200/doy'+format(day+1,'03')
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                np.save(newpath+'/prea'+format(count,'03d')+'.npy', imout[day,:,:])
