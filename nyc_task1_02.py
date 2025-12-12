# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 09:22:15 2024

@author: skrisliu
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import glob
import os
import time
import argparse
import copy
from osgeo import gdal
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data 
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from scipy.stats import pearsonr, gaussian_kde

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.means import Mean
from gpytorch.constraints import Interval


#%% Setup
site = 'nyc'
year = '2023'

MODE = 326     # ---- prefixed code to distinguish task
d,y = 155,2023 # ---- cloudmask
dpre = 91      # ---- predict date

x1,x2 = 550,900
y1,y2 = 650,1150

SPECTRALMODE = 'SEN'   # ---- LDS FOR LANDSAT, SEN FOR SENTINEL2. Default is SEN. 

fp00 = './'     # ---- data path
SAVEPATH = './save/'                # ---- save path


LINEARMEAN = True
SINGLE = True   # GP on single day

batch_size = 2048


#%% cuda
print(torch.__version__)
print('CUDA:',torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#%% load data
# ERA5 in Kelvin
era5lst = np.load(fp00 + site + '/datacube/era5lst_'+year +'.npy')

# spectral images
if SPECTRALMODE=='LDS':
    fp = fp00 + site + '/datacube/' + site + year + 'meanbands.npy'
    ims = np.load(fp)
    # ims = np.transpose(ims,[2,0,1])
    ims = ims/5000
    sen2 = copy.deepcopy(ims)
elif SPECTRALMODE=='SEN':
    fp = fp00 + site + '/datacube/' + site + year + 'meanbands_sen.npy'
    ims = np.load(fp)
    ims = np.transpose(ims,[1,2,0])
    ims = ims/5000
    sen2 = copy.deepcopy(ims)


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


#%% subset
clearmasks = clearmasks[:,x1:x2,y1:y2]
ims = ims[:,x1:x2,y1:y2]
lsts = lsts[:,x1:x2,y1:y2]
era5lst = era5lst[:,x1:x2,y1:y2]

im1z, im1x, im1y = clearmasks.shape

idx1 = np.array(np.where(clearmasks[d]==False)).T   #cloud pixels
idx2 = idx1
clearmasks[dpre,idx2[:,0],idx2[:,1]] = False  # cloud pixels to clouds


### single day
clearmasks = clearmasks[dpre]
lsts = lsts[dpre]

testmask = np.zeros(clearmasks.shape,dtype='bool')
testmask[idx2[:,0],idx2[:,1]] = True    # cloud pixels to clear


#%% load prediction
i = 0
ims = np.zeros([clearmasks.shape[0],clearmasks.shape[1],200] ,dtype=np.float32)
for i in range(200):
    fp = SAVEPATH + site + str(MODE) + '/save200/doy' + format(dpre+1,'03d') + '/prea' + format(i+1,'03d') + '.npy'
    im_ = np.load(fp)
    im_ = im_*0.00341802 + 149.0
    ims[:,:,i] = im_


impre = np.mean(ims,axis=-1)
imres = lsts - impre  # y = yhat + ye
imres = imres.astype(np.float32)


#%% make x train
if True:
    xy1 = np.zeros([imres.shape[0],imres.shape[1],1],dtype=np.float32)
    xy2 = np.zeros([imres.shape[0],imres.shape[1],1],dtype=np.float32)
    for i in range(xy1.shape[0]):
        for j in range(xy1.shape[1]):
            xy1[i,j] = i
            xy2[i,j] = j
        
    sen2 = sen2[x1:x2,y1:y2]
    sen2 = sen2.astype(np.float32)
    im = np.concatenate([sen2,xy1,xy2],axis=-1)
    IMZ = im.shape[-1]
    for i in range(im.shape[-1]):
        im[:,:,i] = ( im[:,:,i]-im[:,:,i].min() ) / ( im[:,:,i].max()-im[:,:,i].min() + 1e-6 )
    im = im.astype(np.float32)
        
    

#%% get train xy
y_train = imres[clearmasks]
x_train = im[clearmasks]

y_test = imres[testmask]
x_test = im[testmask]


#%% linear fit test, get initial parameters
if True:
    RESULT = []
    # pre test
    reg = LinearRegression().fit(x_train, y_train)
    pre0 = reg.predict(x_train)
    mae0 = mean_absolute_error(y_train,pre0)
    rm0 = root_mean_squared_error(y_train,pre0)
    r0 = pearsonr(y_train,pre0)
    RESULT.append([pre0,mae0,rm0,r0.statistic])
    
    model0_params = reg.coef_
    model0_params = model0_params.reshape(-1,1)
    model0_params = np.float32(model0_params)

#%% Define GP. 
if True:
    class LinearMean(Mean):
        def __init__(self, input_size, batch_shape=torch.Size(), bias=True):
            super().__init__()
            self.register_parameter(name="weights", parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size, 1)))
            if bias:
                self.register_parameter(name="bias", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
            else:
                self.bias = None
            self.weights = torch.nn.Parameter(torch.from_numpy(model0_params))
    
        def forward(self, x):
            res = x.matmul(self.weights).squeeze(-1)
            if self.bias is not None:
                res = res + self.bias
            return res
    
    class GP(ApproximateGP):
        def __init__(self, inducing_points,likelihood):
            variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
            variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=False)
            super(GP, self).__init__(variational_strategy)
            if LINEARMEAN:
                self.mean_module = LinearMean(input_size=IMZ)
            else:
                self.mean_module = gpytorch.means.ConstantMean()
            if SINGLE:
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel()) 
            self.likelihood = likelihood
            init_lengthscale = 0.05
            self.covar_module.base_kernel.initialize(lengthscale=init_lengthscale)
    
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    
#%% torch
if True:
    ##### Torch
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    
    ##### train
    np.random.seed(42)
    idx = np.arange(x_train.shape[0])
    np.random.shuffle(idx)
    inducing_points = x_train[idx[:1024], :]
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GP(inducing_points=inducing_points,likelihood=likelihood)
    
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
        x_train = x_train.cuda()
        y_train = y_train.cuda()
    
    
    ### dataloader
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    
    ### ELBO
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y_train.size(0))
    
    
    #%% training
    print('Enter Training #1')
    num_epochs = 20
    
    lr = 0.05
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    likelihood.train()
    for i in np.arange(num_epochs):
        printloss = 0
        count = 0
        for j, (x_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            if SINGLE:
                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    i + 1, num_epochs, loss.item(), 
                    model.covar_module.base_kernel.lengthscale, 
                    model.likelihood.noise.item()
                ))
            else:
                print(i + 1, num_epochs, 
                      format(loss.item(), '.3f'), 
                      format(model.covar_module.base_kernel.lengthscale[0].detach().cpu().numpy()[0], '.3f'), 
                      format(model.covar_module.base_kernel.lengthscale[0].detach().cpu().numpy()[-2], '.3f'), 
                      format(model.covar_module.base_kernel.lengthscale[0].detach().cpu().numpy()[-1], '.3f'),
                      format(model.likelihood.noise.item(), '.3f') )
            loss.backward()
            optimizer.step()
            printloss += loss.item()
            count += 1
        print(i,printloss/count)
    
    #%% training, smaller loss
    if True:
        print('Enter Training #1')
        num_epochs = 5
        
        lr = 0.005
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        model.train()
        likelihood.train()
        for i in np.arange(num_epochs):
            printloss = 0
            count = 0
            for j, (x_batch, y_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(x_batch)
                loss = -mll(output, y_batch)
                if SINGLE:
                    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                        i + 1, num_epochs, loss.item(), 
                        model.covar_module.base_kernel.lengthscale, 
                        model.likelihood.noise.item()
                    ))
                else:
                    print(i + 1, num_epochs, 
                          format(loss.item(), '.3f'), 
                          format(model.covar_module.base_kernel.lengthscale[0].detach().cpu().numpy()[0], '.3f'), 
                          format(model.covar_module.base_kernel.lengthscale[0].detach().cpu().numpy()[-2], '.3f'), 
                          format(model.covar_module.base_kernel.lengthscale[0].detach().cpu().numpy()[-1], '.3f'),
                          format(model.likelihood.noise.item(), '.3f') )
                loss.backward()
                optimizer.step()
                printloss += loss.item()
                count += 1
            print(i,printloss/count)
    
    
    
#%%
TESTPERFORMANCE = True
if TESTPERFORMANCE:

    ###
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 
    
    
    ##### testing
    model.eval()
    likelihood.eval()
    means = torch.tensor([0.])
    lowers = torch.tensor([0.])
    uppers = torch.tensor([0.])
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            preds = model(x_batch.cuda())
            means = torch.cat([means, preds.mean.cpu()]) # only get the mean of the prediction
            
            # std
            lower, upper = preds.confidence_region()
            lower = lower.cpu()
            upper = upper.cpu()
            
            lowers = torch.cat([lowers, lower])
            uppers = torch.cat([uppers, upper])
            
    means = means[1:]
    lowers = lowers[1:]
    uppers = uppers[1:]
    
    # test summary
    pp1 = y_test.numpy()
    pp2 = means.numpy()
    
    r4 = pearsonr(pp1,pp2)
    ma4 = mean_absolute_error(pp1,pp2)
    rm4 = root_mean_squared_error(pp1,pp2)
    print('\nFitting residuals using GP, performance: ')
    print(ma4,rm4,r4)
    
    # if True:
    #     np.save(site+'/resultgp/nyc_doy'+format(doy,'03d')+'_avg_'+dt+'.npy', pp2.reshape([testmask.shape[0], testmask.shape[1]]) )
    #     np.save(site+'/resultgp/nyc_doy'+format(doy,'03d')+'_upp_'+dt+'.npy', uppers.numpy().reshape([testmask.shape[0], testmask.shape[1]]) )
    #     np.save(site+'/resultgp/nyc_doy'+format(doy,'03d')+'_low_'+dt+'.npy', lowers.numpy().reshape([testmask.shape[0], testmask.shape[1]]) )

    if True:
        
        ### ATC
        ypre1 = impre[testmask]
        ypre2 = impre[testmask]
        ygt = lsts[testmask]

        r6 = pearsonr(ygt,ypre2)
        ma6 = mean_absolute_error(ygt,ypre2)
        rm6 = root_mean_squared_error(ygt,ypre2)
        print('\nBaseline result using eATC: ')
        print(ma6,rm6,r6)
        print( np.mean(ygt) - np.mean(ypre2)  )
        
        ### ATC+GP
        ypre1 = impre[testmask]
        ypre2 = impre[testmask] + means.numpy()
        ygt = lsts[testmask]

        r5 = pearsonr(ygt,ypre2)
        ma5 = mean_absolute_error(ygt,ypre2)
        rm5 = root_mean_squared_error(ygt,ypre2)
        print('\nFinal result using eATC+GP [DELAG]: ')
        print(ma5,rm5,r5)
        print( np.mean(ygt) - np.mean(ypre2)  )
        



#%% training data part
if False:   
    model.eval()
    likelihood.eval()
    means = torch.tensor([0.])
    lowers = torch.tensor([0.])
    uppers = torch.tensor([0.])
    ys = torch.tensor([0.])
    with torch.no_grad():
        for x_batch, y_batch in train_loader:
            preds = model(x_batch.cuda())
            means = torch.cat([means, preds.mean.cpu()]) # mean
            ys = torch.cat([ys, y_batch.cpu()])
            
            # std
            lower, upper = preds.confidence_region()
            lower = lower.cpu()
            upper = upper.cpu()
            
            lowers = torch.cat([lowers, lower])
            uppers = torch.cat([uppers, upper])
            
    means = means[1:]
    ys = ys[1:].cpu()
    
    # test summary
    pp1 = ys.numpy()
    pp2 = means.numpy()
    
    r3 = pearsonr(pp1,pp2)
    ma3 = mean_absolute_error(pp1,pp2)
    rm3 = root_mean_squared_error(pp1,pp2)
    print(ma3,rm3,r3)   


#%% test predict
premask = np.ones(testmask.shape,dtype='bool')
x_pre = im[premask]
y_pre = imres[premask]

x_pre = torch.from_numpy(x_pre)
y_pre = torch.from_numpy(y_pre)

pre_dataset = TensorDataset(x_pre, y_pre)
pre_loader = DataLoader(pre_dataset, batch_size=batch_size, shuffle=False) 


# predict
if True:
    model.eval()
    likelihood.eval()
    means = torch.tensor([0.])
    lowers = torch.tensor([0.])
    uppers = torch.tensor([0.])
    with torch.no_grad():
        for x_batch, y_batch in pre_loader:
            preds = model(x_batch.cuda())
            means = torch.cat([means, preds.mean.cpu()]) # mean
            
            # std
            # lower, upper = preds.confidence_region()
            # lower = lower.cpu()
            # upper = upper.cpu()
            
            # lowers = torch.cat([lowers, lower])
            # uppers = torch.cat([uppers, upper])
            
    means = means[1:]
    # lowers = lowers[1:]
    # uppers = uppers[1:]
    ypre99 = means.numpy()

#%%
# Calculate the point density
xy = np.vstack([ygt, ypre2])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so the densest points are plotted on top
idx = z.argsort()
x99, y99, z = ygt[idx], ypre2[idx], z[idx]



#%% plot and save data for plot
def s255(im0, perc=0.5):
    im = copy.deepcopy(im0)
    maxx = np.percentile(im,100-perc)
    minn = np.percentile(im,perc)
    im[im>maxx] = maxx
    im[im<minn] = minn 
    im_new = np.fix((im-minn)/(maxx-minn)*255).astype(np.uint8)
    return im_new

if True:
    d2 = pd.to_datetime(dpre, unit='D', origin=str(year))
    print(d2)
    d3 = format(d2.year,'04d') + format(d2.month,'02d') + format(d2.day,'02d')
    fp = glob.glob(site + '/order/y' + year + '/data' +  d3 + '*.npy')[0]
    im99 = np.load(fp)
    imshow = s255(im99[:,:,[5,4,3]])
 
img1 = imshow[x1:x2, y1:y2, :]
img1b = copy.deepcopy(img1)
img2 = copy.deepcopy(lsts)
img2[testmask] = ypre2
img2b = copy.deepcopy(img2)
img3 = impre + ypre99.reshape(x2-x1,y2-y1)




"""
p1: image, with cloud mask out
p2: original, with cloud mask out
p3: reconstruction, with cloud masks out
p4: scatter plot

"""
cloudmask = np.logical_or(clearmasks,testmask)
cloudmask = cloudmask==False
img1[cloudmask,:] = np.array([255,255,255])
img2[cloudmask] = np.nan


fig = plt.figure(figsize=(11,9),dpi=100)
plt.subplot(221)
plt.imshow(img1b)
plt.xticks([])
plt.yticks([])

plt.subplot(222)
plt.imshow(img3)
plt.xticks([])
plt.yticks([])

plt.subplot(223)
xx = plt.imshow(img2)
ax = plt.gca()
if True:
    cax = inset_axes(ax, width="60%", height="50%", loc='lower left',\
                     bbox_to_anchor=(0.97, 0.75, 0.05, 0.4), bbox_transform=ax.transAxes)
    plt.colorbar(xx, cax=cax)
# plt.colorbar()
plt.xticks([])
plt.yticks([])

plt.subplot(224)
plt.plot([0,999],[0,999], lw=1, c='gray')
# plt.scatter(ygt,ypre2)
plt.scatter(x99, y99, c=z, s=50, cmap='inferno')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Observed LST (K)',fontsize=24)
plt.ylabel('Reconstructed LST (K)',fontsize=24)
plt.xlim(270,320)
plt.ylim(270,320)

plt.tight_layout()
plt.savefig('tmp1.png')
#plt.show()
    
    
    
    
    
#%% save
newpath = SAVEPATH + site + str(MODE) + '/clearRealCloud_'+SPECTRALMODE+'/'
if not os.path.exists(newpath):
    os.makedirs(newpath)

if True:
    np.save(newpath+'im1pre.npy',impre)  # eATC result
    np.save(newpath+'im2pre.npy',impre + ypre99.reshape(x2-x1,y2-y1))  # eATC+GP result
    np.save(newpath+'imshow.npy',imshow) # for visual
    np.save(newpath+'ygt_test.npy',ygt)  # for plot, gt
    np.save(newpath+'ypre2_test.npy',ypre2) # for plot, pre
    np.save(newpath+'cloudmask.npy', cloudmask)
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    























