# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 19:14:48 2024

@author: skrisliu
"""

import numpy as np
import pandas as pd
import glob
import datetime
import matplotlib.pyplot as plt
import copy
from osgeo import gdal
import utm

site = 'nyc'
year = '2023'

#%% Functions
def date2doy(str8):
    dt = datetime.datetime.strptime(str8, '%Y%m%d').date()
    doy = dt.timetuple().tm_yday
    year = str8[:4]
    return doy, year



def s255(im0, perc=0.5):
    im = copy.deepcopy(im0)
    maxx = np.percentile(im,100-perc)
    minn = np.percentile(im,perc)
    im[im>maxx] = maxx
    im[im<minn] = minn 
    im_new = np.fix((im-minn)/(maxx-minn)*255).astype(np.uint8)
    return im_new




#%%
dfs1 = glob.glob(site + '/order/y' + year + '/data*.npy')
dfs2 = glob.glob(site + '/order/y' + year + '/clearmask*.npy')

dfs1 = sorted(dfs1)
dfs2 = sorted(dfs2)



#%% Make datacube holder
im = np.load(dfs2[0])
imx1,imy1 = im.shape

dc1 = np.zeros([365,imx1,imy1],dtype=np.float32)  # LST
dc2 = np.zeros([365,imx1,imy1],dtype='bool')      # clear mask


#%% split. 1733,1667
# 0-400-1100-1733
# 0-750-1667





#%% era5lst
if False:
    x = np.load('nyc/nyc2023lst.npy')
    x = x+273.15
    
    era5lst = np.zeros([365,im.shape[0],im.shape[1]],dtype=np.float32)
    xx1 = im.shape[0]
    yy1 = im.shape[1]
    for i in range(xx1):
        for j in range(yy1):
            era5lst[:,i,j] = x
            
    # plt.plot(era5lst[:,50,50])
    # plt.show()
    if True:
        np.save(site + '/datacube/era5lst_' + year + '.npy', era5lst)

if False:
    fp = site + '/' + site + 'era5_' + year + '.tif'
    era5 = gdal.Open(fp, gdal.GA_ReadOnly)
    geo = era5.GetGeoTransform()
    prj = era5.GetProjection()
    era5 = era5.ReadAsArray()
    
    # fill in mean
    i = 0
    for i in range(365):
        fil = np.nanmean(era5[i,:,:])
        era5[i,:,:][np.isnan(era5[i,:,:])] = fil
    
    xx1 = era5.shape[1]
    yy1 = era5.shape[2]
    era5grid1 = np.zeros([xx1,yy1])
    era5grid2 = np.zeros([xx1,yy1])
    era5grid3 = np.zeros([xx1,yy1])
    era5grid4 = np.zeros([xx1,yy1])
    for i in range(era5grid1.shape[0]):
        for j in range(era5grid1.shape[1]):
            era5grid1[i,j] = geo[3]+geo[5]*i  # lat
            era5grid2[i,j] = geo[0]+geo[1]*j  # lon
            
            xypj = utm.from_latlon(era5grid1[i,j],era5grid2[i,j],force_zone_number=49)
            era5grid3[i,j]  = xypj[1]
            era5grid4[i,j]  = xypj[0]
            
    era5lst = np.zeros([365,im.shape[0],im.shape[1]],dtype=np.float32)
    
    
    ### Landsat latlon
    fp = glob.glob(site + '/clip/y' + year + '/*/*.tif')
    fp = gdal.Open(fp[0],gdal.GA_ReadOnly)
    geo0 = fp.GetGeoTransform()
    prj0 = fp.GetProjection()
    
    xx1 = im.shape[0]
    yy1 = im.shape[1]
    lgrid1 = np.zeros([xx1,yy1])  # lat 
    lgrid2 = np.zeros([xx1,yy1])  # lon
    for i in range(xx1):
        for j in range(yy1):
            lgrid1[i,j] = geo0[3]+geo0[5]*i  # lat
            lgrid2[i,j] = geo0[0]+geo0[1]*j  # lon
    
    
    ### matching
    i = 0
    j = 0
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            loss = (era5grid3-lgrid1[i,j])**2 + (era5grid4-lgrid2[i,j])**2
            nmin = np.argmin(loss)
            # loss.reshape(-1)[nmin]
            # era5grid1.reshape(-1)[nmin]
            # era5grid2.reshape(-1)[nmin]
            era5lst[:,i,j] = era5.reshape([365,-1])[:,nmin]
    
    plt.imshow(era5lst[150,:,:])
    plt.show()
    if True:
        np.save(site + '/datacube/era5lst_' + year + '.npy', era5lst)



#%% Load data

### LST
df = dfs1[0]
for df in dfs1:
    doy,_ = date2doy(df.split('data')[-1][:8])
    im = np.load(df)
    dc1[doy-1,:,:] = im[:,:,0]
    
#%% mannual cloud masking
dc9 = np.zeros([im.shape[0],im.shape[1],9],dtype=np.float32)
DEBUG = 0


#%% manual cloud overwrite 1
# 0-400-1100-1733
# 0-750-1667
x1,x2 = 0,400
y1,y2 = 0,750


### clear mask, data
df = dfs2[0]
ims = []
count = 0
doyloc = []
for df in dfs2:
    doy,_ = date2doy(df.split('clearmask')[-1][:8])
    im = np.load(df)[x1:x2,y1:y2]
    dc2[doy-1,x1:x2,y1:y2] = im
    
    clearp = np.sum(im)/im.shape[0]/im.shape[1]*100
    if clearp>DEBUG:
        print(df)
        str1 = df.split('clearmask')
        str2 = str1[0] + 'data' + str1[1]
        _ = np.load(str2)[x1:x2,y1:y2]
        ims.append(_)
        doyloc.append(doy)
    
        if False:
            # check clear mask
            fig = plt.figure(figsize=(10,4),dpi=150)
            plt.subplot(121)
            plt.imshow(im)
            plt.title(df.split('clearmask')[-1][:8])
            plt.subplot(122)
            plt.imshow(s255(_[:,:,[5,4,3]]))
            plt.title(count)
            plt.show()
            count+=1


if True:
    doyloc2 = np.array(doyloc)
    #overwrite cloud
    
    str1 = '20230121'
    c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
    dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = False
    
    str1 = '20230128'
    c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
    dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = True
    
    str1 = '20230213'
    c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
    dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = True
    
    str1 = '20230214'
    c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
    dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = True
    
    str1 = '20230301'
    c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
    dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = True
    
    str1 = '20230309'
    c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
    dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = True
    
    str1 = '20230310'
    c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
    dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = False
    
    str1 = '20230326'
    c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
    dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = True
    
    str1 = '20230402'
    c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
    dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = True
    
    str1 = '20230403'
    c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
    dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = True
    
    str1 = '20230410'
    c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
    dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = True
    
    str1 = '20230528'
    c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
    dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = True
    
    str1 = '20230529'
    c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
    dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = True
    
    str1 = '20230716'
    c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
    dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = False
    
    str1 = '20230723'
    c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
    dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = True
    
    str1 = '20230901'
    c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
    dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = True
    
    str1 = '20230902'
    c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
    dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = True
    
    str1 = '20230910'
    c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
    dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = False
    
    str1 = '20231003'
    c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
    dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = True
    
    str1 = '20231004'
    c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
    dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = True
    
    str1 = '20231012'
    c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
    dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = True
    
    str1 = '20231020'
    c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
    dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = False
    
    str1 = '20231027'
    c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
    dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = False
    
    str1 = '20231028'
    c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
    dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = True
    
    str1 = '20231105'
    c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
    dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = False
    
    str1 = '20231113'
    c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
    dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = True
    
    str1 = '20231120'
    c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
    dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = True
    
    str1 = '20231215'
    c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
    dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = True
    
    str1 = '20231231'
    c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
    dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = False
    


    
### mean spectral bands
if True:
    ims = np.array(ims,dtype=np.float32)
    ims[ims==0] = 65535
    # dc3 = np.percentile(ims,20,axis=0)
    dc4 = np.sort(ims,axis=0)
    dc3 = np.mean(dc4[0:3,:,:,:],axis=0)
    dc9[x1:x2,y1:y2,:] = dc3
    if False:
        fig = plt.figure(figsize=(8,7),dpi=400)
        plt.imshow(s255(dc3[:,:,[5,4,3]]))
        plt.tight_layout()
        plt.savefig('tmp3.jpg')
        plt.show()



#%% overwrite 2


#%% manual cloud overwrite 2
# 0-400-1100-1733
# 0-750-1667
x1,x2 = 400,1100
y1,y2 = 0,750


### clear mask, data
df = dfs2[0]
ims = []
count = 0
doyloc = []
for df in dfs2:
    doy,_ = date2doy(df.split('clearmask')[-1][:8])
    im = np.load(df)[x1:x2,y1:y2]
    dc2[doy-1,x1:x2,y1:y2] = im
    
    clearp = np.sum(im)/im.shape[0]/im.shape[1]*100
    if clearp>DEBUG:
        print(df)
        str1 = df.split('clearmask')
        str2 = str1[0] + 'data' + str1[1]
        _ = np.load(str2)[x1:x2,y1:y2]
        ims.append(_)
        doyloc.append(doy)
    
        if False:
            # check clear mask
            fig = plt.figure(figsize=(10,4),dpi=150)
            plt.subplot(121)
            plt.imshow(im)
            plt.title(df.split('clearmask')[-1][:8])
            plt.subplot(122)
            plt.imshow(s255(_[:,:,[5,4,3]]))
            plt.title(count)
            plt.show()
            count+=1


if True:
    doyloc2 = np.array(doyloc)
    #overwrite cloud
    
    # False, cloudy
    str1a = ['20230104', '20230105', '20230310', '20230427', '20230716', '20230824', 
             '20230910', '20230918', '20231020', '20231230']
    
    # True, clear
    str1b = ['20230128', '20230214', '20230318', '20230326', '20230402', '20230403', 
             '20230410', '20230529', '20230809', '20230901', '20230902', '20230917', 
             '20231003', '20231004', '20231012', '20231113', '20231120', '20231215']
    
    
    
    for str1 in str1a:
        c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
        dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = False
    
    for str1 in str1b:
        c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
        dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = True
    
    
### mean spectral bands
if True:
    ims = np.array(ims,dtype=np.float32)
    ims[ims==0] = 65535
    # dc3 = np.percentile(ims,20,axis=0)
    dc4 = np.sort(ims,axis=0)
    dc3 = np.mean(dc4[0:3,:,:,:],axis=0)
    dc9[x1:x2,y1:y2,:] = dc3
    if False:
        fig = plt.figure(figsize=(8,7),dpi=400)
        plt.imshow(s255(dc3[:,:,[5,4,3]]))
        plt.tight_layout()
        plt.savefig('tmp3.jpg')
        plt.show()


#%% overwrite 3


#%% manual cloud overwrite 3
# 0-400-1100-1733
# 0-750-1667
x1,x2 = 1100,1733
y1,y2 = 0,750


### clear mask, data
df = dfs2[0]
ims = []
count = 0
doyloc = []
for df in dfs2:
    doy,_ = date2doy(df.split('clearmask')[-1][:8])
    im = np.load(df)[x1:x2,y1:y2]
    dc2[doy-1,x1:x2,y1:y2] = im
    
    clearp = np.sum(im)/im.shape[0]/im.shape[1]*100
    if clearp>DEBUG:
        print(df)
        str1 = df.split('clearmask')
        str2 = str1[0] + 'data' + str1[1]
        _ = np.load(str2)[x1:x2,y1:y2]
        ims.append(_)
        doyloc.append(doy)
    
        if False:
            # check clear mask
            fig = plt.figure(figsize=(10,4),dpi=150)
            plt.subplot(121)
            plt.imshow(im)
            plt.title(df.split('clearmask')[-1][:8])
            plt.subplot(122)
            plt.imshow(s255(_[:,:,[5,4,3]]))
            plt.title(count)
            plt.show()
            count+=1


if True:
    doyloc2 = np.array(doyloc)
    #overwrite cloud
    
    # False, cloudy
    str1a = ['20230104', '20230105', '20230310', '20230317', '20230427', '20230716', 
             '20230909', '20230910', '20230918', '20231104', '20231105', '20231112', 
             '20231222', '20231230']
    
    # True, clear
    str1b = ['20230128', '20230214', '20230318', '20230326', '20230402', '20230403', 
             '20230410', '20230411', '20230528', '20230529', '20230901', '20230902', '20230917', 
             '20231003', '20231004', '20231012', '20231120', '20231215']
    
    
    
    for str1 in str1a:
        c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
        dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = False
    
    for str1 in str1b:
        c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
        dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = True
    
    
### mean spectral bands
if True:
    ims = np.array(ims,dtype=np.float32)
    ims[ims==0] = 65535
    # dc3 = np.percentile(ims,20,axis=0)
    dc4 = np.sort(ims,axis=0)
    dc3 = np.mean(dc4[0:3,:,:,:],axis=0)
    dc9[x1:x2,y1:y2,:] = dc3
    if False:
        fig = plt.figure(figsize=(8,7),dpi=400)
        plt.imshow(s255(dc3[:,:,[5,4,3]]))
        plt.tight_layout()
        plt.savefig('tmp3.jpg')
        plt.show()



#%% overwrite 4


#%% manual cloud overwrite 4
# 0-400-1100-1733
# 0-750-1667
x1,x2 = 0, 400
y1,y2 = 750, 1667


### clear mask, data
df = dfs2[0]
ims = []
count = 0
doyloc = []
for df in dfs2:
    doy,_ = date2doy(df.split('clearmask')[-1][:8])
    im = np.load(df)[x1:x2,y1:y2]
    dc2[doy-1,x1:x2,y1:y2] = im
    
    clearp = np.sum(im)/im.shape[0]/im.shape[1]*100
    if clearp>DEBUG:
        print(df)
        str1 = df.split('clearmask')
        str2 = str1[0] + 'data' + str1[1]
        _ = np.load(str2)[x1:x2,y1:y2]
        ims.append(_)
        doyloc.append(doy)
    
        if False:
            # check clear mask
            fig = plt.figure(figsize=(10,4),dpi=150)
            plt.subplot(121)
            plt.imshow(im)
            plt.title(df.split('clearmask')[-1][:8])
            plt.subplot(122)
            plt.imshow(s255(_[:,:,[5,4,3]]))
            plt.title(count)
            plt.show()
            count+=1


if True:
    doyloc2 = np.array(doyloc)
    #overwrite cloud
    
    # False, cloudy
    str1a = ['20230105', '20230205', '20230426', '20230504', 
             '20230716',  
             '20230909', '20230910', '20230918', '20231027', '20231104', '20231105', 
             '20231112', '20231129', '20231206', 
             '20231222', '20231230', '20231231']
    
    # True, clear
    str1b = ['20230128', '20230214', '20230318', '20230326', '20230402', '20230403', 
             '20230410', '20230528', '20230529', '20230809', '20230901', '20230902', '20230917', 
             '20231003', '20231004', '20231012', '20231019', '20231113', '20231215']
    
    
    
    for str1 in str1a:
        c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
        dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = False
    
    for str1 in str1b:
        c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
        dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = True
    
    
### mean spectral bands
if True:
    ims = np.array(ims,dtype=np.float32)
    ims[ims==0] = 65535
    # dc3 = np.percentile(ims,20,axis=0)
    dc4 = np.sort(ims,axis=0)
    dc3 = np.mean(dc4[0:3,:,:,:],axis=0)
    dc9[x1:x2,y1:y2,:] = dc3
    if False:
        fig = plt.figure(figsize=(8,7),dpi=400)
        plt.imshow(s255(dc3[:,:,[5,4,3]]))
        plt.tight_layout()
        plt.savefig('tmp3.jpg')
        plt.show()


#%% overwrite 5


#%% manual cloud overwrite 5
# 0-400-1100-1733
# 0-750-1667
x1,x2 = 400,1100
y1,y2 = 750,1667


### clear mask, data
df = dfs2[0]
ims = []
count = 0
doyloc = []
for df in dfs2:
    doy,_ = date2doy(df.split('clearmask')[-1][:8])
    im = np.load(df)[x1:x2,y1:y2]
    dc2[doy-1,x1:x2,y1:y2] = im
    
    clearp = np.sum(im)/im.shape[0]/im.shape[1]*100
    if clearp>DEBUG:
        print(df)
        str1 = df.split('clearmask')
        str2 = str1[0] + 'data' + str1[1]
        _ = np.load(str2)[x1:x2,y1:y2]
        ims.append(_)
        doyloc.append(doy)
    
        if False:
            # check clear mask
            fig = plt.figure(figsize=(10,4),dpi=150)
            plt.subplot(121)
            plt.imshow(im)
            plt.title(df.split('clearmask')[-1][:8])
            plt.subplot(122)
            plt.imshow(s255(_[:,:,[5,4,3]]))
            plt.title(count)
            plt.show()
            count+=1


if True:
    doyloc2 = np.array(doyloc)
    #overwrite cloud
    
    # False, cloudy
    str1a = ['20230105', '20230120', '20230121', '20230206', '20230427', '20230504', 
             '20230512', '20230614', '20230724', '20230909', '20230910', '20230918', 
             '20231104', '20231112', '20231214', '20231231']
    
    # True, clear
    str1b = ['20230128', '20230301', '20230309', '20230326', '20230402', '20230410', 
             '20230529', '20230606', '20230723', '20230731', '20230902', '20230917', 
             '20231003', '20231113', '20231120']
    
    
    
    for str1 in str1a:
        c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
        dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = False
    
    for str1 in str1b:
        c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
        dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = True
    
    
### mean spectral bands
if True:
    ims = np.array(ims,dtype=np.float32)
    ims[ims==0] = 65535
    # dc3 = np.percentile(ims,20,axis=0)
    dc4 = np.sort(ims,axis=0)
    dc3 = np.mean(dc4[0:3,:,:,:],axis=0)
    dc9[x1:x2,y1:y2,:] = dc3
    if False:
        fig = plt.figure(figsize=(8,7),dpi=400)
        plt.imshow(s255(dc3[:,:,[5,4,3]]))
        plt.tight_layout()
        plt.savefig('tmp3.jpg')
        plt.show()



#%% overwrite 6


#%% manual cloud overwrite 6
# 0-400-1100-1733
# 0-750-1667
x1,x2 = 1100, 1733
y1,y2 = 750, 1667


### clear mask, data
df = dfs2[0]
ims = []
count = 0
doyloc = []
for df in dfs2:
    doy,_ = date2doy(df.split('clearmask')[-1][:8])
    im = np.load(df)[x1:x2,y1:y2]
    dc2[doy-1,x1:x2,y1:y2] = im
    
    clearp = np.sum(im)/im.shape[0]/im.shape[1]*100
    if clearp>DEBUG:
        print(df)
        str1 = df.split('clearmask')
        str2 = str1[0] + 'data' + str1[1]
        _ = np.load(str2)[x1:x2,y1:y2]
        ims.append(_)
        doyloc.append(doy)
    
        if False:
            # check clear mask
            fig = plt.figure(figsize=(10,4),dpi=150)
            plt.subplot(121)
            plt.imshow(im)
            plt.title(df.split('clearmask')[-1][:8])
            plt.subplot(122)
            plt.imshow(s255(_[:,:,[5,4,3]]))
            plt.title(count)
            plt.show()
            count+=1


if True:
    doyloc2 = np.array(doyloc)
    #overwrite cloud
    
    # False, cloudy
    str1a = ['20230105', '20230120', '20230205', '20230213', '20230310', '20230317', 
             '20230427', '20230504', '20230512', '20230707', '20230716', '20230909', 
             '20230910', '20230918', '20231104', '20231105', '20231222', '20231230', 
             '20231231']
    
    # True, clear
    str1b = ['20230128', '20230214', '20230301', '20230309', '20230326', '20230402', 
             '20230403', '20230410', '20230521', '20230529', '20230606', '20230708', 
             '20230731', '20230901', '20230902', '20230917', '20231003', '20231004', 
             '20231011', '20231019', '20231028', '20231120', '20231215']
    
    
    
    for str1 in str1a:
        c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
        dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = False
    
    for str1 in str1b:
        c1 = np.where(doyloc2==date2doy(str1)[0])[0][0]
        dc2[date2doy(str1)[0]-1,x1:x2,y1:y2][np.sum(ims[c1][:,:,[5,4,3]],axis=-1)!=0] = True
    
    
### mean spectral bands
if True:
    ims = np.array(ims,dtype=np.float32)
    ims[ims==0] = 65535
    # dc3 = np.percentile(ims,20,axis=0)
    dc4 = np.sort(ims,axis=0)
    dc3 = np.mean(dc4[0:3,:,:,:],axis=0)
    dc9[x1:x2,y1:y2,:] = dc3
    
    if False:
        fig = plt.figure(figsize=(8,7),dpi=400)
        plt.imshow(s255(dc3[:,:,[5,4,3]]))
        plt.tight_layout()
        plt.savefig('tmp3.jpg')
        plt.show()




#%% ending





#%% save data
if True:
    np.save(site + '/datacube/' + site + year + 'meanbands.npy', dc9[:,:,[1,2,3,4,5,6,7]])
    np.save(site + '/datacube/' + site + year + 'clearmasks.npy', dc2)
    np.save(site + '/datacube/' + site + year + 'lsts.npy', dc1)





























