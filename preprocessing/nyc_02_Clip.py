# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 23:57:09 2024

@author: skrisliu
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import os
from osgeo import gdal
# import rasterio
# from rasterio.warp import calculate_default_transform, reproject, Resampling
# 

site = 'nyc'
year = '2023'


def setGeo(geotransform,bgx,bgy):
    reset0 = geotransform[0] + bgx*geotransform[1]
    reset3 = geotransform[3] + bgy*geotransform[5]
    reset = (reset0,geotransform[1],geotransform[2],
             reset3,geotransform[4],geotransform[5])
    return reset


#%% espg? no


#%%
sfile = site + '/clip/y' + year + '/'
fp = site + '/unzip/y' + year + '/' + '/uz_*L2SP_*'
dfss = glob.glob(fp)



if True:
    x1,x2 = 560000, 610010 # --- 1667
    y1,y2 = 4532000, 4480010  # --- 1733



#%%
each = dfss[17]
for each in dfss:
    print(each)
    dfs = glob.glob(each+'/*.TIF')
    df = dfs[5]
    count = 0
    for df in dfs:
        im = gdal.Open(df,gdal.GA_ReadOnly)
        prj = im.GetProjection()
        geo = im.GetGeoTransform()
        
        x1b = np.uint32((x1-geo[0])/geo[1])
        x2b = np.uint32((x2-geo[0])/geo[1])
        
        y1b = np.uint32((y1-geo[3])/geo[5])
        y2b = np.uint32((y2-geo[3])/geo[5])
        
        im = im.GetRasterBand(1).ReadAsArray()
        imnew = im[y1b:y2b,x1b:x2b]
        
        newgeo = setGeo(geo,x1b,y1b)
        
        # save as geocode-tif
        im1x,im1y = imnew.shape
        x = df.split('\\')[-1].split('_')
        path = sfile + 't' + x[3] + '_' + x[2] + '_' + x[0]
        name = sfile + 't' + x[3] + '_' + x[2] + '_' + x[0] + '/t' + x[3] + '_' + x[-2] + '_' + x[-1]
        
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        
        np.save(name[:-4]+'.npy',imnew)
        
        outdata = gdal.GetDriverByName('GTiff').Create(name, im1y, im1x, 1, gdal.GDT_UInt16, [ 'COMPRESS=LZW' ])
        outdata.SetGeoTransform(newgeo)
        outdata.SetProjection(prj)
        outdata.GetRasterBand(1).WriteArray(imnew)
        outdata.FlushCache() ##saves to disk!!
        outdata = None   
        





