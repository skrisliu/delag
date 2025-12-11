# -*- coding: utf-8 -*-
"""

@author: skrisliu

Order and Masking

"""

import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy


site = 'nyc'
year = '2023'

def int_to_16bit_binary(num):
    """Converts an integer to a 16-bit binary string."""
    return format(num, '016b')
# 21824: 0101010101000000  # inverse order
# 22080: 0101011001000000

def s255(im0, perc=0.5):
    im = copy.deepcopy(im0)
    maxx = np.percentile(im,100-perc)
    minn = np.percentile(im,perc)
    im[im>maxx] = maxx
    im[im<minn] = minn 
    im_new = np.fix((im-minn)/(maxx-minn)*255).astype(np.uint8)
    return im_new


#%%
fps = glob.glob(site + '/clip\y'+year+'/*')


#%%
fp0 = fps[4]
for fp0 in fps:
    name = fp0.split('\\')[-1]
    doy = name.split('_')[0][1:]
    fp1s = sorted(glob.glob(fp0+'\\*.npy'))
    fp1 = fp1s[0]
    ims = []
    for i in [11,3,4,5,6,7,8,9,1]:
        im = np.load(fp1s[i])
        ims.append(im)
    ims = np.array(ims)
    ims = np.transpose(ims,[1,2,0])   # B10, B1, B2, B3, B4, B5, B6, B7, QA
    path = site + '/order/y' + year
    
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
        
    name2 = path + '/data'+doy+'_'+name.split('_')[-1]+'.npy'
    np.save(name2, ims)
        
    ## official cloud/clear mask
    codes = np.unique(ims[:,:,-1])
    clear = []
    for code in codes:
        code_ = int_to_16bit_binary(code)
        if code_[9]=='1':
            clear.append(code)
    cmask = np.zeros(im.shape,dtype='bool')
    for c_ in clear:  # clearmask
        cmask[ims[:,:,-1]==c_] = True 
     
    name3 = path + '/clearmask'+doy+'_'+name.split('_')[-1]+'.npy'
    np.save(name3, cmask)








#%%