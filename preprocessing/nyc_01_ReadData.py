# -*- coding: utf-8 -*-
"""
@author: skrisliu

nyc

"""

import numpy as np
import matplotlib.pyplot as plt
import tarfile
import glob

site = 'nyc'
year = '2023'

#%%
dfs = glob.glob(site + '/zip/y'+year+'/*.tar')


#%%
df = dfs[0]
for df in dfs:
    with tarfile.open(df, "r") as tf:
        print("Opened tarfile")
        tf.extractall(path=site+"/unzip/y"+year+"/uz_" + df.split('\\')[1][:-4])
        print("All files extracted")
    
    
    
    
    
    
    
    
    
    
    
    
    
























