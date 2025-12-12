# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 07:54:56 2024

@author: skrisliu
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



site = 'nyc'
year = '2023'


MODE = 326   ### ---- 326=nyc_clear. 
d,y = 155,2023 ### --------------------------- cloudmask
dpre = 91  ### --- predict date

SPECTRALMODE = 'SEN'   ## LDS FOR LANDSAT, SEN FOR SENTINEL2

fp00 = '/mnt/external/MTL5/t20241222_DELAG_tgrs/'
SAVEPATH = '/mnt/external/MTL7/t250524_DELAG/data/'


### cuda
print(torch.__version__)
print('CUDA:',torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# visualize RS data to RGB figure
def s255(im0, perc=0.5):
    im = copy.deepcopy(im0)
    maxx = np.percentile(im,100-perc)
    minn = np.percentile(im,perc)
    im[im>maxx] = maxx
    im[im<minn] = minn 
    im_new = np.fix((im-minn)/(maxx-minn)*255).astype(np.uint8)
    return im_new


#%%
# ERA5 in Kelvin
era5lst = np.load(site + '/datacube/era5lst_'+year +'.npy')

# spectral images
if SPECTRALMODE=='LDS':
    fp = fp00 + site + '/datacube/' + site + year + 'meanbands.npy'
    ims = np.load(fp)
    ims = np.transpose(ims,[2,0,1])
    ims = ims/5000
elif SPECTRALMODE=='SEN':
    fp = fp00 + '/demdata/' + 'sen2'+site+'4.tif'
    ims = gdal.Open(fp, gdal.GA_ReadOnly)
    ims = ims.ReadAsArray()
    ims = ims/5000

# LST in Kelvin
fp = fp00 + site + '/datacube/' + site + year + 'lsts.npy'
lsts = np.load(fp)
mask = lsts==0
lsts = lsts*0.00341802 + 149.0
lsts[mask] = 0

# cloud masks
fp = fp00 + site + '/datacube/' + site + year + 'clearmasks.npy'
clearmasks = np.load(fp)
im1z, im1x, im1y = clearmasks.shape


#%% Get Random 90% Clouds --- not using, using clear-cloud day pair, real world clouds for testing
x1,x2 = 550,900
y1,y2 = 650,1150
if False:
    plt.imshow(sen2[x1:x2,y1:y2,2])
    plt.show()

i = 0
ps = []
for i in range(365):
    _ = clearmasks[i,x1:x2,y1:y2]
    p_ = np.sum(_) / (_.shape[0]*_.shape[1])
    ps.append([i,p_])
    if p_!=0:
        print([i,p_])  # [91, 1.0],  [155, 0.47936666666666666]
        
### visual
if False:
    d,y = 155,2023 ### --------------------------- cloudmask
    dpre = 91  ### --- predict date
    d2 = pd.to_datetime(d, unit='D', origin=str(year))
    print(d2)
    d3 = format(d2.year,'04d') + format(d2.month,'02d') + format(d2.day,'02d')
    fp = glob.glob(site + '/order/y' + year + '/data' +  d3 + '*.npy')[0]
    im = np.load(fp)
    # im = im
    
    imshow = s255(im[:,:,[5,4,3]])
    imshow = imshow[x1:x2,y1:y2,:]
    fig = plt.figure(figsize=(6,5),dpi=100)
    plt.imshow(imshow)
    plt.tight_layout()
    plt.show()

    imask = clearmasks[d][x1:x2,y1:y2]==False
    imshow[imask,:] = np.array([255,255,0])
    fig = plt.figure(figsize=(6,5),dpi=100)
    plt.imshow(imshow)
    plt.tight_layout()
    plt.show()


#%% new masking subset
clearmasks = clearmasks[:,x1:x2,y1:y2]
ims = ims[:,x1:x2,y1:y2]
lsts = lsts[:,x1:x2,y1:y2]
era5lst = era5lst[:,x1:x2,y1:y2]

im1z, im1x, im1y = clearmasks.shape

idx1 = np.array(np.where(clearmasks[d]==False)).T   ### ---- cloud patterns from d
# np.random.seed(68)
# np.random.shuffle(idx1)
# n1 = idx1.shape[0]//5
# idx2 = idx1[:n1,:]
idx2 = idx1
clearmasks[dpre,idx2[:,0],idx2[:,1]] = False  #### predict date, make sure on the predict date, it has the cloud patten the same from d
if False:
    plt.imshow(clearmasks[dpre])
if True:
    newpath = site + '/train' + str(MODE)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    np.save(newpath + '/' + site + '_mask'+str(MODE)+'_idx' + str(d) + '.npy', clearmasks)


#%%
IMZ = ims.shape[0]


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
ims = ims.reshape([1,IMZ,im1x,im1y])
ims = torch.from_numpy(ims)
clearmasks = torch.from_numpy(clearmasks)
lsts = torch.from_numpy(lsts)

## to gpu
era5lst = era5lst.to(device)
ims = ims.to(device)
clearmasks = clearmasks.to(device)
lsts = lsts.to(device)


#%% build model
model = atcnet(imz=IMZ)
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
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    
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
optimizer = optim.Adam(model.parameters(), lr=0.1)
count = 0
for i in range(800):
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
                newpath = SAVEPATH+site+str(MODE)+'/save100/doy'+format(day+1,'03')
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                np.save(newpath+'/prea'+format(count,'03d')+'.npy', imout[day,:,:])


































































