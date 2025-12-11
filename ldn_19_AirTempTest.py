# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 01:55:50 2024
Released on 20251211

@author: skrisliu

London stations = 6
03779                              London Weather Centre
03672                                           Northolt
03772                            London Heathrow Airport
EGLC0                                London / Abbey Wood
EGKB0                        Biggin Hill / berry's green
03781                                             Kenley
EGTI0                          Leavesden / North Watford   ---- not available

"""



#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from meteostat import Stations
from meteostat import Point, Daily
from datetime import datetime
from osgeo import gdal
import utm
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.linear_model import LinearRegression
import pvlib 
from pvlib.location import Location
from scipy.stats import pearsonr
# from matplotlib import rc

# rc('text',usetex=True)
# rc('text.latex', preamble=r'\usepackage{color}')



'''
SET STATION AND YEAR
'''
site = 'ldn'
year = '2023'

BIGX = []
BIGY = []
BIGMASK = []


#%% Get Stations
stations = Stations()
stations = stations.nearby(51.485, -0.170)
station = stations.fetch(50)
if False:
    station.to_csv(site+'/'+site+'_stations.csv')
codes = ['03779', '03672', '03772', 'EGLC0', 'EGKB0', '03781', 'EGTI0']

start = datetime(2023, 1, 1)
end = datetime(2023, 12, 31)


'''
LOAD DEM AND NDVI DATA
'''
#%% load dem
demdata = gdal.Open('demdata/dem_ldn4.tif', gdal.GA_ReadOnly)
demdata = demdata.ReadAsArray()
dems = []

im = gdal.Open('demdata/sen2ldn4.tif', gdal.GA_ReadOnly)
ndvi = im.ReadAsArray()
ndvi = np.float32(ndvi)
ndvi = (ndvi[3,:,:] - ndvi[2,:,:])  / (ndvi[3,:,:] + ndvi[2,:,:])
ndvis = []



'''
EACH STATION'S DATA IS PRE-SAVED ALREADY
'''
#%% 
if False:
    n = 0
    data = Daily(station[station.index==codes[n]], start, end)
    data = data.fetch()
    print(data)
    data.to_pickle(site+'/'+site+year+'_'+codes[n]+'.pkl')

if False:
    n = 1
    data = Daily(station[station.index==codes[n]], start, end)
    data = data.fetch()
    print(data)
    data.to_pickle(site+'/'+site+year+'_'+codes[n]+'.pkl')

if False:
    n = 2
    data = Daily(station[station.index==codes[n]], start, end)
    data = data.fetch()
    print(data)
    data.to_pickle(site+'/'+site+year+'_'+codes[n]+'.pkl')
    
if False:
    n = 3
    data = Daily(station[station.index==codes[n]], start, end)
    data = data.fetch()
    print(data)
    data.to_pickle(site+'/'+site+year+'_'+codes[n]+'.pkl')

if False:
    n = 4
    data = Daily(station[station.index==codes[n]], start, end)
    data = data.fetch()
    print(data)
    data.to_pickle(site+'/'+site+year+'_'+codes[n]+'.pkl')

if False:
    n = 5
    data = Daily(station[station.index==codes[n]], start, end)
    data = data.fetch()
    print(data)
    data.to_pickle(site+'/'+site+year+'_'+codes[n]+'.pkl')

if False:
    n = 6
    data = Daily(station[station.index==codes[n]], start, end)
    data = data.fetch()
    print(data)
    data.to_pickle(site+'/'+site+year+'_'+codes[n]+'.pkl')






'''
THE IMAGE IS TOO LARGE, SPLIT INTO 9 PARTS. 
LOADING ALL 9 PARTS. 
'''
#%%
# split, 1573,2053
# 0-520-1040-1573
# 0-700-1400-2053
im0 = np.zeros([365,1573,2053],dtype=np.float32)



#%% ldn1
site = 'ldn1'
df = site + '/pre1/' + site + '1avg.npy'
im = np.load(df)
im = im*0.00341802 + 149.0
if True:
    x1,x2 = 0,520
    y1,y2 = 0,700
im0[:,x1:x2,y1:y2] = im
print(im.min(),im.max())

#%% ldn2
site = 'ldn2'
df = site + '/pre1/' + site + '1avg.npy'
im = np.load(df)
im = im*0.00341802 + 149.0
if True:
    x1,x2 = 520,1040
    y1,y2 = 0,700
im0[:,x1:x2,y1:y2] = im
print(im.min(),im.max())


#%% ldn3
site = 'ldn3'
df = site + '/pre1/' + site + '1avg.npy'
im = np.load(df)
im = im*0.00341802 + 149.0
if True:
    x1,x2 = 1040,1573
    y1,y2 = 0,700
im0[:,x1:x2,y1:y2] = im
print(im.min(),im.max())

#%% ldn4
site = 'ldn4'
df = site + '/pre1/' + site + '1avg.npy'
im = np.load(df)
im = im*0.00341802 + 149.0
if True:
    x1,x2 = 0,520
    y1,y2 = 700,1400
im0[:,x1:x2,y1:y2] = im
print(im.min(),im.max())

#%% ldn5
site = 'ldn5'
df = site + '/pre1/' + site + '1avg.npy'
im = np.load(df)
im = im*0.00341802 + 149.0
if True:
    x1,x2 = 520,1040
    y1,y2 = 700,1400
im0[:,x1:x2,y1:y2] = im
print(im.min(),im.max())

#%% ldn6
site = 'ldn6'
df = site + '/pre1/' + site + '1avg.npy'
im = np.load(df)
im = im*0.00341802 + 149.0
if True:
    x1,x2 = 1040,1573
    y1,y2 = 700,1400
im0[:,x1:x2,y1:y2] = im
print(im.min(),im.max())

#%% ldn7
site = 'ldn7'
df = site + '/pre1/' + site + '1avg.npy'
im = np.load(df)
im = im*0.00341802 + 149.0
if True:
    x1,x2 = 0,520
    y1,y2 = 1400,2053
im0[:,x1:x2,y1:y2] = im
print(im.min(),im.max())

#%% ldn8
site = 'ldn8'
df = site + '/pre1/' + site + '1avg.npy'
im = np.load(df)
im = im*0.00341802 + 149.0
if True:
    x1,x2 = 520,1040
    y1,y2 = 1400,2053
im0[:,x1:x2,y1:y2] = im
print(im.min(),im.max())

#%% ldn9
site = 'ldn9'
df = site + '/pre1/' + site + '1avg.npy'
im = np.load(df)
im = im*0.00341802 + 149.0
if True:
    x1,x2 = 1040,1573
    y1,y2 = 1400,2053
im0[:,x1:x2,y1:y2] = im
print(im.min(),im.max())


#%%
fig = plt.figure(figsize=(8,6),dpi=200)
plt.imshow(im0[300,:,:],vmin=270,vmax=310)
plt.colorbar()
plt.tight_layout()
plt.show()


#%% geo prj
im = gdal.Open('ldn/t20230109_SR_B1.TIF', gdal.GA_ReadOnly)
geo = im.GetGeoTransform()
prj = im.GetProjection()



'''
MATCH EACH STATION AND REGRESSION DATA. 
'''


#%% Get UTM, station 03779
df = pd.read_csv('ldn/ldn_stations.csv')
stat = df[df['id']=='03779']
lat = stat['latitude'].values[0]
lon = stat['longitude'].values[0]

utmx = utm.from_latlon(latitude=lat, longitude=lon, force_zone_number=31)
x0 = int((utmx[0]-geo[0]) / geo[1])
y0 = int((utmx[1]-geo[3]) / geo[5])
y_pre0 = im0[:,y0,x0]

df_gt = pd.read_pickle('ldn/ldn2023_03779.pkl')
y_gt = df_gt['tavg'].values + 273.15

### validate 01
np.mean(y_gt) - np.mean(y_pre0)         
mean_absolute_error(y_gt, y_pre0)        
root_mean_squared_error(y_gt, y_pre0)   

### reg 1
y = y_gt
x1 = y_pre0
x = np.array([x1]).T
reg = LinearRegression().fit(x,y)
reg.score(x, y)
reg.coef_
reg.intercept_
y_pre2 = reg.predict(x)

np.mean(y_gt) - np.mean(y_pre2)         
mean_absolute_error(y_gt, y_pre2)        
root_mean_squared_error(y_gt, y_pre2)   


### reg 2
y = y_gt
x1 = y_pre0
x2 = np.cos(np.pi*2/365*(np.arange(365)+1 - 218) )
x = np.array([x1,x2]).T

reg = LinearRegression().fit(x,y)
reg.score(x, y)
reg.coef_
reg.intercept_
y_pre3 = reg.predict(x)

np.mean(y_gt) - np.mean(y_pre3)         
mean_absolute_error(y_gt, y_pre3)        
root_mean_squared_error(y_gt, y_pre3)   


### reg 3, solar
days = [31,28,31,30,31,30, 31,31,30,31,30,31]
location = Location(lat, lon)

sza = []
for i in range(12):
    for j in range(days[i]):
        now = datetime(2023,i+1,j+1,10,51)
        solar_position = location.get_solarposition(now)
        sza.append(solar_position['zenith'].values[0])
sza = np.array(sza)
sza2 = np.cos(np.pi*sza/180)

y = y_gt
x1 = y_pre0
x2 = np.cos(np.pi*2/365*(np.arange(365)+1 - 218) )
x3 = sza2
x = np.array([x1,x2,x3]).T

reg = LinearRegression().fit(x,y)
reg.score(x, y)
reg.coef_
reg.intercept_     # 1.02340594,  -0.25485692, -15.24746422, -3.1729773494292317
y_pre4 = reg.predict(x)

np.mean(y_gt) - np.mean(y_pre4)         
mean_absolute_error(y_gt, y_pre4)        
root_mean_squared_error(y_gt, y_pre4)  


BIGX.append(x)
BIGY.append(y)

### clear date
fp = 'ldn/datacube/ldn2023clearmasks.npy'
clear = np.load(fp)
y_clear = clear[:,y0,x0]


BIGMASK.append(y_clear)


### dem data
dems.append(np.ones([365,])*demdata[y0,x0])
ndvis.append(np.ones([365,]) *ndvi[y0,x0] )


if False:
    fig = plt.figure(figsize=(7,5),dpi=200)
    plt.plot(np.arange(365)+1, y_gt, c='blue', label='gt')
    plt.plot(np.arange(365)+1, y_pre4, c='red', label='pre')
    plt.legend()
    plt.tight_layout()
    plt.show()
 

if False:
    fig = plt.figure(figsize=(4,3),dpi=200)
    plt.scatter(y_gt,y_pre4,s=60,facecolors='none',color='red',marker='o')
    plt.xlabel('gt')
    plt.ylabel('pre')
    plt.xlim(260,315)
    plt.ylim(260,315)
    plt.tight_layout()
    plt.show()
    
### print clear, cloudy, no obs
# mae=0.65, rmse=0.81
if False:
    fig = plt.figure(figsize=(3.5,3), dpi=300)
    plt.plot([-50,1000],[-50,1000],color='black',lw=1)
    plt.scatter(y_gt[~y_clear],y_pre4[~y_clear],s=60,facecolors='none',color='blue',marker='o',label='cloudy, no obs')
    plt.scatter(y_gt[y_clear],y_pre4[y_clear],s=60,facecolors='none',color='red',marker='v',label='clear')
    plt.xlabel('gt')
    plt.ylabel('pre')
    plt.xlim(260,315)
    plt.ylim(260,315)
    plt.legend()
    plt.tight_layout()
    plt.show()
    


'''
STATION #2
'''


#%% Station 2. Get UTM, station 03672
scode = '03672'
df = pd.read_csv('ldn/ldn_stations.csv')
stat = df[df['id']==scode]
lat = stat['latitude'].values[0]
lon = stat['longitude'].values[0]

utmx = utm.from_latlon(latitude=lat, longitude=lon, force_zone_number=31)
x0 = int((utmx[0]-geo[0]) / geo[1])
y0 = int((utmx[1]-geo[3]) / geo[5])
y_pre0 = im0[:,y0,x0]

df_gt = pd.read_pickle('ldn/ldn2023_'+scode+'.pkl')
y_gt = df_gt['tavg'].values + 273.15

### validate 01
np.mean(y_gt) - np.mean(y_pre0)         
mean_absolute_error(y_gt, y_pre0)        
root_mean_squared_error(y_gt, y_pre0)   

### reg 1
y = y_gt
x1 = y_pre0
x = np.array([x1]).T
reg = LinearRegression().fit(x,y)
reg.score(x, y)
reg.coef_
reg.intercept_
y_pre2 = reg.predict(x)

np.mean(y_gt) - np.mean(y_pre2)         
mean_absolute_error(y_gt, y_pre2)        
root_mean_squared_error(y_gt, y_pre2)   


### reg 2
y = y_gt
x1 = y_pre0
x2 = np.cos(np.pi*2/365*(np.arange(365)+1 - 218) )
x = np.array([x1,x2]).T

reg = LinearRegression().fit(x,y)
reg.score(x, y)
reg.coef_
reg.intercept_
y_pre3 = reg.predict(x)

np.mean(y_gt) - np.mean(y_pre3)         
mean_absolute_error(y_gt, y_pre3)        
root_mean_squared_error(y_gt, y_pre3)   


### reg 3, solar
days = [31,28,31,30,31,30, 31,31,30,31,30,31]
location = Location(lat, lon)

sza = []
for i in range(12):
    for j in range(days[i]):
        now = datetime(2023,i+1,j+1,10,51)
        solar_position = location.get_solarposition(now)
        sza.append(solar_position['zenith'].values[0])
sza = np.array(sza)
sza2 = np.cos(np.pi*sza/180)

y = y_gt
x1 = y_pre0
x2 = np.cos(np.pi*2/365*(np.arange(365)+1 - 218) )
x3 = sza2
x = np.array([x1,x2,x3]).T

reg = LinearRegression().fit(x,y)
reg.score(x, y)
reg.coef_
reg.intercept_     # 1.02340594,  -0.25485692, -15.24746422, -3.1729773494292317
y_pre4 = reg.predict(x)

np.mean(y_gt) - np.mean(y_pre4)         
mean_absolute_error(y_gt, y_pre4)        
root_mean_squared_error(y_gt, y_pre4)  


BIGX.append(x)
BIGY.append(y)


### clear date
fp = 'ldn/datacube/ldn2023clearmasks.npy'
clear = np.load(fp)
y_clear = clear[:,y0,x0]

BIGMASK.append(y_clear)

### dem data
dems.append(np.ones([365,])*demdata[y0,x0])
ndvis.append(np.ones([365,]) *ndvi[y0,x0] )



if False:
    fig = plt.figure(figsize=(7,5),dpi=200)
    plt.plot(np.arange(365)+1, y_gt, c='blue', label='gt')
    plt.plot(np.arange(365)+1, y_pre4, c='red', label='pre')
    plt.legend()
    plt.tight_layout()
    plt.show()
 

if False:
    fig = plt.figure(figsize=(4,3),dpi=200)
    plt.scatter(y_gt,y_pre4,s=60,facecolors='none',color='blue',marker='o')
    plt.xlabel('gt')
    plt.ylabel('pre')
    plt.xlim(260,315)
    plt.ylim(260,315)
    plt.tight_layout()
    plt.show()
    
### print clear, cloudy, no obs
# mae=0.609, rmse=0.797
if False:
    fig = plt.figure(figsize=(3.5,3), dpi=300)
    plt.plot([-50,1000],[-50,1000],color='black',lw=1)
    plt.scatter(y_gt[~y_clear],y_pre4[~y_clear],s=60,facecolors='none',color='blue',marker='o',label='cloudy, no obs')
    plt.scatter(y_gt[y_clear],y_pre4[y_clear],s=60,facecolors='none',color='red',marker='v',label='clear')
    plt.xlabel('gt')
    plt.ylabel('pre')
    plt.xlim(260,315)
    plt.ylim(260,315)
    plt.legend()
    plt.tight_layout()
    plt.show()



'''
STATION #3
'''

#%% Station 3

#%% Station 3. Get UTM, station 03772
scode = '03772'
df = pd.read_csv('ldn/ldn_stations.csv')
stat = df[df['id']==scode]
lat = stat['latitude'].values[0]
lon = stat['longitude'].values[0]

utmx = utm.from_latlon(latitude=lat, longitude=lon, force_zone_number=31)
x0 = int((utmx[0]-geo[0]) / geo[1])
y0 = int((utmx[1]-geo[3]) / geo[5])
y_pre0 = im0[:,y0,x0]

df_gt = pd.read_pickle('ldn/ldn2023_'+scode+'.pkl')
y_gt = df_gt['tavg'].values + 273.15

### validate 01
np.mean(y_gt) - np.mean(y_pre0)         
mean_absolute_error(y_gt, y_pre0)        
root_mean_squared_error(y_gt, y_pre0)   

### reg 1
y = y_gt
x1 = y_pre0
x = np.array([x1]).T
reg = LinearRegression().fit(x,y)
reg.score(x, y)
reg.coef_
reg.intercept_
y_pre2 = reg.predict(x)

np.mean(y_gt) - np.mean(y_pre2)         
mean_absolute_error(y_gt, y_pre2)        
root_mean_squared_error(y_gt, y_pre2)   


### reg 2
y = y_gt
x1 = y_pre0
x2 = np.cos(np.pi*2/365*(np.arange(365)+1 - 218) )
x = np.array([x1,x2]).T

reg = LinearRegression().fit(x,y)
reg.score(x, y)
reg.coef_
reg.intercept_
y_pre3 = reg.predict(x)

np.mean(y_gt) - np.mean(y_pre3)         
mean_absolute_error(y_gt, y_pre3)        
root_mean_squared_error(y_gt, y_pre3)   


### reg 3, solar
days = [31,28,31,30,31,30, 31,31,30,31,30,31]
location = Location(lat, lon)

sza = []
for i in range(12):
    for j in range(days[i]):
        now = datetime(2023,i+1,j+1,10,51)
        solar_position = location.get_solarposition(now)
        sza.append(solar_position['zenith'].values[0])
sza = np.array(sza)
sza2 = np.cos(np.pi*sza/180)

y = y_gt
x1 = y_pre0
x2 = np.cos(np.pi*2/365*(np.arange(365)+1 - 218) )
x3 = sza2
x = np.array([x1,x2,x3]).T

reg = LinearRegression().fit(x,y)
reg.score(x, y)
reg.coef_
reg.intercept_     # 1.02340594,  -0.25485692, -15.24746422, -3.1729773494292317
y_pre4 = reg.predict(x)

np.mean(y_gt) - np.mean(y_pre4)         
mean_absolute_error(y_gt, y_pre4)        
root_mean_squared_error(y_gt, y_pre4)  


BIGX.append(x)
BIGY.append(y)


### clear date
fp = 'ldn/datacube/ldn2023clearmasks.npy'
clear = np.load(fp)
y_clear = clear[:,y0,x0]

BIGMASK.append(y_clear)

### dem data
dems.append(np.ones([365,])*demdata[y0,x0])
ndvis.append(np.ones([365,]) *ndvi[y0,x0] )


if False:
    fig = plt.figure(figsize=(7,5),dpi=200)
    plt.plot(np.arange(365)+1, y_gt, c='blue', label='gt')
    plt.plot(np.arange(365)+1, y_pre4, c='red', label='pre')
    plt.legend()
    plt.tight_layout()
    plt.show()
 

if False:
    fig = plt.figure(figsize=(4,3),dpi=200)
    plt.scatter(y_gt,y_pre4,s=60,facecolors='none',color='blue',marker='o')
    plt.xlabel('gt')
    plt.ylabel('pre')
    plt.xlim(260,315)
    plt.ylim(260,315)
    plt.tight_layout()
    plt.show()
    
### print clear, cloudy, no obs
# mae=0.609, rmse=0.797
if False:
    fig = plt.figure(figsize=(3.5,3), dpi=300)
    plt.plot([-50,1000],[-50,1000],color='black',lw=1)
    plt.scatter(y_gt[~y_clear],y_pre4[~y_clear],s=60,facecolors='none',color='blue',marker='o',label='cloudy, no obs')
    plt.scatter(y_gt[y_clear],y_pre4[y_clear],s=60,facecolors='none',color='red',marker='v',label='clear')
    plt.xlabel('gt')
    plt.ylabel('pre')
    plt.xlim(260,315)
    plt.ylim(260,315)
    plt.legend()
    plt.tight_layout()
    plt.show()




'''
STATION #4
'''

#%% Station 4


#%% Station 4. Get UTM, station EGLC0
scode = 'EGLC0'
df = pd.read_csv('ldn/ldn_stations.csv')
stat = df[df['id']==scode]
lat = stat['latitude'].values[0]
lon = stat['longitude'].values[0]

utmx = utm.from_latlon(latitude=lat, longitude=lon, force_zone_number=31)
x0 = int((utmx[0]-geo[0]) / geo[1])
y0 = int((utmx[1]-geo[3]) / geo[5])
y_pre0 = im0[:,y0,x0]

df_gt = pd.read_pickle('ldn/ldn2023_'+scode+'.pkl')
y_gt = df_gt['tavg'].values + 273.15

### validate 01
np.mean(y_gt) - np.mean(y_pre0)         
mean_absolute_error(y_gt, y_pre0)        
root_mean_squared_error(y_gt, y_pre0)   

### reg 1
y = y_gt
x1 = y_pre0
x = np.array([x1]).T
reg = LinearRegression().fit(x,y)
reg.score(x, y)
reg.coef_
reg.intercept_
y_pre2 = reg.predict(x)

np.mean(y_gt) - np.mean(y_pre2)         
mean_absolute_error(y_gt, y_pre2)        
root_mean_squared_error(y_gt, y_pre2)   


### reg 2
y = y_gt
x1 = y_pre0
x2 = np.cos(np.pi*2/365*(np.arange(365)+1 - 218) )
x = np.array([x1,x2]).T

reg = LinearRegression().fit(x,y)
reg.score(x, y)
reg.coef_
reg.intercept_
y_pre3 = reg.predict(x)

np.mean(y_gt) - np.mean(y_pre3)         
mean_absolute_error(y_gt, y_pre3)        
root_mean_squared_error(y_gt, y_pre3)   


### reg 3, solar
days = [31,28,31,30,31,30, 31,31,30,31,30,31]
location = Location(lat, lon)

sza = []
for i in range(12):
    for j in range(days[i]):
        now = datetime(2023,i+1,j+1,10,51)
        solar_position = location.get_solarposition(now)
        sza.append(solar_position['zenith'].values[0])
sza = np.array(sza)
sza2 = np.cos(np.pi*sza/180)

y = y_gt
x1 = y_pre0
x2 = np.cos(np.pi*2/365*(np.arange(365)+1 - 218) )
x3 = sza2
x = np.array([x1,x2,x3]).T

reg = LinearRegression().fit(x,y)
reg.score(x, y)
reg.coef_
reg.intercept_     # 1.02340594,  -0.25485692, -15.24746422, -3.1729773494292317
y_pre4 = reg.predict(x)

np.mean(y_gt) - np.mean(y_pre4)         
mean_absolute_error(y_gt, y_pre4)        
root_mean_squared_error(y_gt, y_pre4)  


BIGX.append(x)
BIGY.append(y)


### clear date
fp = 'ldn/datacube/ldn2023clearmasks.npy'
clear = np.load(fp)
y_clear = clear[:,y0,x0]

BIGMASK.append(y_clear)


### dem data
dems.append(np.ones([365,])*demdata[y0,x0])
ndvis.append(np.ones([365,]) *ndvi[y0,x0] )


if False:
    fig = plt.figure(figsize=(7,5),dpi=200)
    plt.plot(np.arange(365)+1, y_gt, c='blue', label='gt')
    plt.plot(np.arange(365)+1, y_pre4, c='red', label='pre')
    plt.legend()
    plt.tight_layout()
    plt.show()
 

if False:
    fig = plt.figure(figsize=(4,3),dpi=200)
    plt.scatter(y_gt,y_pre4,s=60,facecolors='none',color='blue',marker='o')
    plt.xlabel('gt')
    plt.ylabel('pre')
    plt.xlim(260,315)
    plt.ylim(260,315)
    plt.tight_layout()
    plt.show()
    
### print clear, cloudy, no obs
# mae=0.609, rmse=0.797
if False:
    fig = plt.figure(figsize=(3.5,3), dpi=300)
    plt.plot([-50,1000],[-50,1000],color='black',lw=1)
    plt.scatter(y_gt[~y_clear],y_pre4[~y_clear],s=60,facecolors='none',color='blue',marker='o',label='cloudy, no obs')
    plt.scatter(y_gt[y_clear],y_pre4[y_clear],s=60,facecolors='none',color='red',marker='v',label='clear')
    plt.xlabel('gt')
    plt.ylabel('pre')
    plt.xlim(260,315)
    plt.ylim(260,315)
    plt.legend()
    plt.tight_layout()
    plt.show()







'''
STATION #5
'''

#%% Station 5


#%% Station 5. Get UTM, station EGKB0
scode = 'EGKB0'
df = pd.read_csv('ldn/ldn_stations.csv')
stat = df[df['id']==scode]
lat = stat['latitude'].values[0]
lon = stat['longitude'].values[0]

utmx = utm.from_latlon(latitude=lat, longitude=lon, force_zone_number=31)
x0 = int((utmx[0]-geo[0]) / geo[1])
y0 = int((utmx[1]-geo[3]) / geo[5])
y_pre0 = im0[:,y0,x0]

df_gt = pd.read_pickle('ldn/ldn2023_'+scode+'.pkl')
y_gt = df_gt['tavg'].values + 273.15

### validate 01
np.mean(y_gt) - np.mean(y_pre0)         
mean_absolute_error(y_gt, y_pre0)        
root_mean_squared_error(y_gt, y_pre0)   

### reg 1
y = y_gt
x1 = y_pre0
x = np.array([x1]).T
reg = LinearRegression().fit(x,y)
reg.score(x, y)
reg.coef_
reg.intercept_
y_pre2 = reg.predict(x)

np.mean(y_gt) - np.mean(y_pre2)         
mean_absolute_error(y_gt, y_pre2)        
root_mean_squared_error(y_gt, y_pre2)   


### reg 2
y = y_gt
x1 = y_pre0
x2 = np.cos(np.pi*2/365*(np.arange(365)+1 - 218) )
x = np.array([x1,x2]).T

reg = LinearRegression().fit(x,y)
reg.score(x, y)
reg.coef_
reg.intercept_
y_pre3 = reg.predict(x)

np.mean(y_gt) - np.mean(y_pre3)         
mean_absolute_error(y_gt, y_pre3)        
root_mean_squared_error(y_gt, y_pre3)   


### reg 3, solar
days = [31,28,31,30,31,30, 31,31,30,31,30,31]
location = Location(lat, lon)

sza = []
for i in range(12):
    for j in range(days[i]):
        now = datetime(2023,i+1,j+1,10,51)
        solar_position = location.get_solarposition(now)
        sza.append(solar_position['zenith'].values[0])
sza = np.array(sza)
sza2 = np.cos(np.pi*sza/180)

y = y_gt
x1 = y_pre0
x2 = np.cos(np.pi*2/365*(np.arange(365)+1 - 218) )
x3 = sza2
x = np.array([x1,x2,x3]).T

reg = LinearRegression().fit(x,y)
reg.score(x, y)
reg.coef_
reg.intercept_     # 1.02340594,  -0.25485692, -15.24746422, -3.1729773494292317
y_pre4 = reg.predict(x)

np.mean(y_gt) - np.mean(y_pre4)         
mean_absolute_error(y_gt, y_pre4)        
root_mean_squared_error(y_gt, y_pre4)  


BIGX.append(x)
BIGY.append(y)


### clear date
fp = 'ldn/datacube/ldn2023clearmasks.npy'
clear = np.load(fp)
y_clear = clear[:,y0,x0]

BIGMASK.append(y_clear)

### dem data
dems.append(np.ones([365,])*demdata[y0,x0])
ndvis.append(np.ones([365,]) *ndvi[y0,x0] )



if False:
    fig = plt.figure(figsize=(7,5),dpi=200)
    plt.plot(np.arange(365)+1, y_gt, c='blue', label='gt')
    plt.plot(np.arange(365)+1, y_pre4, c='red', label='pre')
    plt.legend()
    plt.tight_layout()
    plt.show()
 

if False:
    fig = plt.figure(figsize=(4,3),dpi=200)
    plt.scatter(y_gt,y_pre4,s=60,facecolors='none',color='blue',marker='o')
    plt.xlabel('gt')
    plt.ylabel('pre')
    plt.xlim(260,315)
    plt.ylim(260,315)
    plt.tight_layout()
    plt.show()
    
### print clear, cloudy, no obs
# mae=0.609, rmse=0.797
if False:
    fig = plt.figure(figsize=(3.5,3), dpi=300)
    plt.plot([-50,1000],[-50,1000],color='black',lw=1)
    plt.scatter(y_gt[~y_clear],y_pre4[~y_clear],s=60,facecolors='none',color='blue',marker='o',label='cloudy, no obs')
    plt.scatter(y_gt[y_clear],y_pre4[y_clear],s=60,facecolors='none',color='red',marker='v',label='clear')
    plt.xlabel('gt')
    plt.ylabel('pre')
    plt.xlim(260,315)
    plt.ylim(260,315)
    plt.legend()
    plt.tight_layout()
    plt.show()
    






'''
STATION #6
'''

#%% Station 6

#%% Station 6. Get UTM, station 03781
scode = '03781'
df = pd.read_csv('ldn/ldn_stations.csv')
stat = df[df['id']==scode]
lat = stat['latitude'].values[0]
lon = stat['longitude'].values[0]

utmx = utm.from_latlon(latitude=lat, longitude=lon, force_zone_number=31)
x0 = int((utmx[0]-geo[0]) / geo[1])
y0 = int((utmx[1]-geo[3]) / geo[5])
y_pre0 = im0[:,y0,x0]

df_gt = pd.read_pickle('ldn/ldn2023_'+scode+'.pkl')
y_gt = df_gt['tavg'].values + 273.15

### validate 01
np.mean(y_gt) - np.mean(y_pre0)         
mean_absolute_error(y_gt, y_pre0)        
root_mean_squared_error(y_gt, y_pre0)   

### reg 1
y = y_gt
x1 = y_pre0
x = np.array([x1]).T
reg = LinearRegression().fit(x,y)
reg.score(x, y)
reg.coef_
reg.intercept_
y_pre2 = reg.predict(x)

np.mean(y_gt) - np.mean(y_pre2)         
mean_absolute_error(y_gt, y_pre2)        
root_mean_squared_error(y_gt, y_pre2)   


### reg 2
y = y_gt
x1 = y_pre0
x2 = np.cos(np.pi*2/365*(np.arange(365)+1 - 218) )
x = np.array([x1,x2]).T

reg = LinearRegression().fit(x,y)
reg.score(x, y)
reg.coef_
reg.intercept_
y_pre3 = reg.predict(x)

np.mean(y_gt) - np.mean(y_pre3)         
mean_absolute_error(y_gt, y_pre3)        
root_mean_squared_error(y_gt, y_pre3)   


### reg 3, solar
days = [31,28,31,30,31,30, 31,31,30,31,30,31]
location = Location(lat, lon)

sza = []
for i in range(12):
    for j in range(days[i]):
        now = datetime(2023,i+1,j+1,10,51)
        solar_position = location.get_solarposition(now)
        sza.append(solar_position['zenith'].values[0])
sza = np.array(sza)
sza2 = np.cos(np.pi*sza/180)

y = y_gt
x1 = y_pre0
x2 = np.cos(np.pi*2/365*(np.arange(365)+1 - 218) )
x3 = sza2
x = np.array([x1,x2,x3]).T

reg = LinearRegression().fit(x,y)
reg.score(x, y)
reg.coef_
reg.intercept_     # 1.02340594,  -0.25485692, -15.24746422, -3.1729773494292317
y_pre4 = reg.predict(x)

np.mean(y_gt) - np.mean(y_pre4)         
mean_absolute_error(y_gt, y_pre4)        
root_mean_squared_error(y_gt, y_pre4)  


BIGX.append(x)
BIGY.append(y)


### clear date
fp = 'ldn/datacube/ldn2023clearmasks.npy'
clear = np.load(fp)
y_clear = clear[:,y0,x0]

BIGMASK.append(y_clear)


### dem data
dems.append(np.ones([365,])*demdata[y0,x0])
ndvis.append(np.ones([365,]) *ndvi[y0,x0] )



if False:
    fig = plt.figure(figsize=(7,5),dpi=200)
    plt.plot(np.arange(365)+1, y_gt, c='blue', label='gt')
    plt.plot(np.arange(365)+1, y_pre4, c='red', label='pre')
    plt.legend()
    plt.tight_layout()
    plt.show()
 

if False:
    fig = plt.figure(figsize=(4,3),dpi=200)
    plt.scatter(y_gt,y_pre4,s=60,facecolors='none',color='blue',marker='o')
    plt.xlabel('gt')
    plt.ylabel('pre')
    plt.xlim(260,315)
    plt.ylim(260,315)
    plt.tight_layout()
    plt.show()
    
### print clear, cloudy, no obs
# mae=0.609, rmse=0.797
if False:
    fig = plt.figure(figsize=(3.5,3), dpi=300)
    plt.plot([-50,1000],[-50,1000],color='black',lw=1)
    plt.scatter(y_gt[~y_clear],y_pre4[~y_clear],s=60,facecolors='none',color='blue',marker='o',label='cloudy, no obs')
    plt.scatter(y_gt[y_clear],y_pre4[y_clear],s=60,facecolors='none',color='red',marker='v',label='clear')
    plt.xlabel('gt')
    plt.ylabel('pre')
    plt.xlim(260,315)
    plt.ylim(260,315)
    plt.legend()
    plt.tight_layout()
    plt.show()





'''
RUN ALL DATA VIA ONE REGRESSION. 
'''


#%% TOTAL
dems2 = np.concatenate(dems)
ndvis2 = np.concatenate(ndvis)

BIGX2 = np.concatenate(BIGX)
BIGX2 = np.concatenate([BIGX2,dems2.reshape(-1,1), ndvis2.reshape(-1,1)],axis=1)
BIGY2 = np.concatenate(BIGY)
BIGMASK2 = np.concatenate(BIGMASK)

reg = LinearRegression().fit(BIGX2,BIGY2)
reg.score(BIGX2, BIGY2)
reg.coef_
reg.intercept_
y_pre9 = reg.predict(BIGX2)

np.mean(BIGY2) - np.mean(y_pre9)         
mean_absolute_error(BIGY2, y_pre9)        
root_mean_squared_error(BIGY2, y_pre9) 
pearsonr(BIGY2, y_pre9)

me1 = mean_absolute_error(BIGY2, y_pre9)    
me2 = root_mean_squared_error(BIGY2, y_pre9) 
me3 = pearsonr(BIGY2, y_pre9)
text = 'MAE='+format(me1,'.2f') + '\nRMSE=' + format(me2, '.2f') + '\nR$^2$=' + format(me3.statistic**2, '.2f')
print(text)

if False:
    fig = plt.figure(figsize=(4,3),dpi=200)
    plt.scatter(BIGY2,y_pre9,s=60,facecolors='none',color='blue',marker='o')
    plt.xlabel('gt')
    plt.ylabel('pre')
    plt.xlim(260,315)
    plt.ylim(260,315)
    plt.tight_layout()
    plt.show()

if False:
    n2 = np.sum(BIGMASK2)
    n1 = BIGMASK2.shape[0] - n2
    fig = plt.figure(figsize=(3.5,3), dpi=300)
    plt.plot([-50,1000],[-50,1000],color='black',lw=1)
    plt.scatter(BIGY2[~BIGMASK2],y_pre9[~BIGMASK2],s=60,facecolors='none',color='blue',marker='o',label='Cloudy/NoObs, n=' + str(n1))
    plt.scatter(BIGY2[BIGMASK2],y_pre9[BIGMASK2],s=60,facecolors='none',color='red',marker='v',label='Clear, n=' + str(n2))
    plt.xlabel('gt')
    plt.ylabel('pre')
    plt.xlim(260,315)
    plt.ylim(260,315)
    plt.text(313,262,text,fontsize=8,horizontalalignment='right')
    plt.legend()
    plt.tight_layout()
    plt.show()  





'''
RUN REGRESSSIONS SEPARATELY FOR CLEAR AND CLOUDY/NO OBS
'''

#%% seperate fit
import matplotlib

me11 = []


### clear
x2a = BIGX2[BIGMASK2]
y2a = BIGY2[BIGMASK2]

reg1 = LinearRegression().fit(x2a,y2a)
reg1.coef_
reg1.intercept_
ypre2a = reg1.predict(x2a)

_ = np.mean(y2a) - np.mean(ypre2a)    
me11.append(_)     
_ = mean_absolute_error(y2a, ypre2a)      
me11.append(_)     
_ = root_mean_squared_error(y2a, ypre2a) 
me11.append(_)   
_ = pearsonr(y2a, ypre2a).statistic**2
me11.append(_)   




### cloudy/no obs
me12 = []

x2b = BIGX2[~BIGMASK2]
y2b = BIGY2[~BIGMASK2]

reg2 = LinearRegression().fit(x2b,y2b)
reg2.coef_
reg2.intercept_
ypre2b = reg2.predict(x2b)

_ = np.mean(y2b) - np.mean(ypre2b)   
me12.append(_)       
_ = mean_absolute_error(y2b, ypre2b)  
me12.append(_)       
_ = root_mean_squared_error(y2b, ypre2b) 
me12.append(_) 
_ = pearsonr(y2b, ypre2b).statistic**2
me12.append(_)  

y11 = reg1.predict(BIGX2)
y12 = reg2.predict(BIGX2)


if False:
    n2 = np.sum(BIGMASK2)
    n1 = BIGMASK2.shape[0] - n2
    fig = plt.figure(figsize=(3.5,3), dpi=300)
    plt.plot([-50,1000],[-50,1000],color='black',lw=1)
    plt.scatter(BIGY2[~BIGMASK2],y_pre9[~BIGMASK2],s=60,facecolors='none',color='blue',marker='o',label='Cloud-Covered or\nNo Landsat Overpass')
    plt.scatter(BIGY2[BIGMASK2],y_pre9[BIGMASK2],s=60,facecolors='none',color='red',marker='v',label='Valid-Observed LST')
    plt.xlabel('gt')
    plt.ylabel('pre')
    plt.xlim(260,315)
    plt.ylim(260,315)
    plt.text(313,262,text,fontsize=7,horizontalalignment='right')
    plt.legend(loc=2,fontsize=8)
    plt.tight_layout()
    plt.show()  

# plot
if False:

    
    xlabel = 'Observed T$_{air}$ (K)'
    ylabel = 'Estimated T$_{air}$ (K)'
    x1lim,x2lim = 260,320
    text = 'MAE='+format(me1,'.2f') + '\nRMSE=' + format(me2, '.2f') + '\nR$^2$=' + format(me3.statistic**2, '.2f')
    
    n2 = np.sum(BIGMASK2)
    n1 = BIGMASK2.shape[0] - n2
    
    
    fig = plt.figure(figsize=(3.5,3), dpi=300)
    plt.plot([-50,1000],[-50,1000],color='black',lw=1)
    ax = plt.gca()
    
    sc1 = ax.scatter(BIGY2[~BIGMASK2],y_pre9[~BIGMASK2],s=60,facecolors='none',color='blue',marker='o',label=' ')
    legend1 = ax.legend(handles=[sc1], loc=2,fontsize=8,frameon=False,bbox_to_anchor=(-0.03, 1.00))
    
    sc2 = ax.scatter(BIGY2[BIGMASK2],y_pre9[BIGMASK2],s=60,facecolors='none',color='red',marker='v',label=' ')
    legend2 = plt.legend(handles=[sc2],loc=4,fontsize=8,frameon=False,bbox_to_anchor=(0.77, 0.165))
    
    plt.xlabel('gt')
    plt.ylabel('pre')
    plt.plot([-50,1000],[-50,1000],color='black',lw=1)
    ax = plt.gca()

    txt1 = 'R$^2$='+format(me11[3],'.2f')+'\nRMSE='+ format(me11[2],'.2f') + '\nMAE=' + format(me11[1],'.2f')    
    txt2 = 'R$^2$='+format(me12[3],'.2f')+'\nRMSE='+ format(me12[2],'.2f') + '\nMAE=' + format(me12[1],'.2f')  

    plt.text(266, 318.5, "Cloud-Covered or\nNo Landsat Overpass", fontsize=8,horizontalalignment='left',verticalalignment='top')

    plt.text(262, 312, txt1, fontsize=9,horizontalalignment='left',verticalalignment='top')
    
    plt.text(318, 272, "Clear-Sky LST", fontsize=8, horizontalalignment='right',verticalalignment='bottom')

    plt.text(318, 261, txt2, fontsize=9,horizontalalignment='right',verticalalignment='bottom')
    ax = plt.gca()
    ax.add_artist(legend1)
    plt.text(265, 261, codes[idx1], fontsize=8, horizontalalignment='left',verticalalignment='bottom')
    
    plt.tight_layout()
    plt.show()  
    





'''
PLOT EACH STATION AS SUBPLOT
ALL STATION TOGETHER
'''

#%% seperately plot
"""
03779 EGRB London Weather Centre
03672 EGWU Northolt
03772 EGLL London Heathrow Airport
EGLC0 EGLC London / Abbey Wood
EGKB0 EGKB Biggin Hill / berry's green
07381 KLY Kenley 
"""

### clear
m21 = [] # clear sky
data21 = [] # clear data

### cloudy / no obs
m22 = [] # cloudy sky
data22 = []  #cloudy data

bias22 = []
mae22 = []
rmse22 = []
rr22 = []

bias21 = []
mae21 = []
rmse21 = []
rr21 = []



for n11 in np.arange(0,2190,365):
    n12 = n11+365
    a1 = y11[n11:n12][BIGMASK2[n11:n12]]            # prediction, clear sky
    a2 = y12[n11:n12][~BIGMASK2[n11:n12]]           # prediction, cloud cover
    a1b = BIGY2[n11:n12][BIGMASK2[n11:n12]]         # gt, clearsky 
    a2b = BIGY2[n11:n12][~BIGMASK2[n11:n12]]        # gt, cloud cover
    ###
    _ = np.mean(a1b) - np.mean(a1)    
    bias21.append(_)     
    _ = mean_absolute_error(a1b, a1)      
    mae21.append(_)     
    _ = root_mean_squared_error(a1b, a1) 
    rmse21.append(_)   
    _ = pearsonr(a1b, a1).statistic**2
    rr21.append(_)   
    data21.append([a1b,a1])
    
    
    _ = np.mean(a2b) - np.mean(a2)    
    bias22.append(_)     
    _ = mean_absolute_error(a2b, a2)      
    mae22.append(_)     
    _ = root_mean_squared_error(a2b, a2) 
    rmse22.append(_)   
    _ = pearsonr(a2b, a2).statistic**2
    rr22.append(_)   
    data22.append([a2b,a2])



#%% paper plot
codes = ['EGRB', 'EGWU', 'EGLL', 'EGLC', 'EGKB', 'KLY']

if True:
    xlabel = 'Observed T$_{air}$ (K)'
    ylabel = 'Estimated T$_{air}$ (K)'
    x1lim,x2lim = 260,310
    x1lim,x2lim = 265,305
    n2 = np.sum(BIGMASK2)
    n1 = BIGMASK2.shape[0] - n2
    fig = plt.figure(figsize=(7,9), dpi=300)
    plt.subplot(321)
    
    idx1 = 0
    plt.plot([-50,1000],[-50,1000],color='black',lw=1)
    ax = plt.gca()
    sc1 = ax.scatter(data22[idx1][0],data22[idx1][1],s=60,facecolors='none',color='blue',marker='o',label=' ')
    legend1 = ax.legend(handles=[sc1], loc=2,fontsize=8,frameon=False,bbox_to_anchor=(-0.03, 1.00))
    sc2 = plt.scatter(data21[idx1][0],data21[idx1][1],s=60,facecolors='none',color='red',marker='v',label=' ')
    legend2 = plt.legend(handles=[sc2],loc=4,fontsize=8,frameon=False,bbox_to_anchor=(0.77, 0.165))
    plt.text(0.11, 0.98, "Cloud-Covered or\nNo Landsat Overpass", fontsize=8,horizontalalignment='left',verticalalignment='top',transform=ax.transAxes)
    txt1 = 'R$^2$='+format(rr22[idx1],'.2f')+'\nRMSE='+ format(rmse22[idx1],'.2f') + ' K\nMAE=' + format(mae22[idx1],'.2f') + ' K'
    plt.text(0.04, 0.86, txt1, fontsize=9,horizontalalignment='left',verticalalignment='top',transform=ax.transAxes)
    
    plt.text(0.98, 0.205, "Clear-Sky LST", fontsize=8, horizontalalignment='right',verticalalignment='bottom',transform=ax.transAxes)
    txt2 = 'R$^2$='+format(rr21[idx1],'.2f')+'\nRMSE='+ format(rmse21[idx1],'.2f') + ' K\nMAE=' + format(mae21[idx1],'.2f') + ' K'
    plt.text(0.98, 0.02, txt2, fontsize=9,horizontalalignment='right',verticalalignment='bottom',transform=ax.transAxes)
    ax = plt.gca()
    ax.add_artist(legend1)
    plt.text(0.1, 0.025, codes[idx1], fontsize=8, horizontalalignment='left',verticalalignment='bottom',transform=ax.transAxes)
    
    plt.xlim(x1lim,x2lim)
    plt.ylim(x1lim,x2lim)
    plt.ylabel(ylabel)

    

    ###### sub 2
    plt.subplot(322)
    idx1 = 1
    plt.plot([-50,1000],[-50,1000],color='black',lw=1)
    ax = plt.gca()
    sc1 = ax.scatter(data22[idx1][0],data22[idx1][1],s=60,facecolors='none',color='blue',marker='o',label=' ')
    legend1 = ax.legend(handles=[sc1], loc=2,fontsize=8,frameon=False,bbox_to_anchor=(-0.03, 1.00))
    sc2 = plt.scatter(data21[idx1][0],data21[idx1][1],s=60,facecolors='none',color='red',marker='v',label=' ')
    legend2 = plt.legend(handles=[sc2],loc=4,fontsize=8,frameon=False,bbox_to_anchor=(0.77, 0.165))
    plt.text(0.11, 0.98, "Cloud-Covered or\nNo Landsat Overpass", fontsize=8,horizontalalignment='left',verticalalignment='top',transform=ax.transAxes)
    txt1 = 'R$^2$='+format(rr22[idx1],'.2f')+'\nRMSE='+ format(rmse22[idx1],'.2f') + ' K\nMAE=' + format(mae22[idx1],'.2f') + ' K'
    plt.text(0.04, 0.86, txt1, fontsize=9,horizontalalignment='left',verticalalignment='top',transform=ax.transAxes)
    
    plt.text(0.98, 0.205, "Clear-Sky LST", fontsize=8, horizontalalignment='right',verticalalignment='bottom',transform=ax.transAxes)
    txt2 = 'R$^2$='+format(rr21[idx1],'.2f')+'\nRMSE='+ format(rmse21[idx1],'.2f') + ' K\nMAE=' + format(mae21[idx1],'.2f') + ' K'
    plt.text(0.98, 0.02, txt2, fontsize=9,horizontalalignment='right',verticalalignment='bottom',transform=ax.transAxes)
    ax = plt.gca()
    ax.add_artist(legend1)
    plt.text(0.1, 0.025, codes[idx1], fontsize=8, horizontalalignment='left',verticalalignment='bottom',transform=ax.transAxes)
    
    plt.xlim(x1lim,x2lim)
    plt.ylim(x1lim,x2lim)
    
    
    
    
    
    
    ### sub 3
    plt.subplot(323)
    idx1 = 2
    plt.plot([-50,1000],[-50,1000],color='black',lw=1)
    ax = plt.gca()
    sc1 = ax.scatter(data22[idx1][0],data22[idx1][1],s=60,facecolors='none',color='blue',marker='o',label=' ')
    legend1 = ax.legend(handles=[sc1], loc=2,fontsize=8,frameon=False,bbox_to_anchor=(-0.03, 1.00))
    sc2 = plt.scatter(data21[idx1][0],data21[idx1][1],s=60,facecolors='none',color='red',marker='v',label=' ')
    legend2 = plt.legend(handles=[sc2],loc=4,fontsize=8,frameon=False,bbox_to_anchor=(0.77, 0.165))
    plt.text(0.11, 0.98, "Cloud-Covered or\nNo Landsat Overpass", fontsize=8,horizontalalignment='left',verticalalignment='top',transform=ax.transAxes)
    txt1 = 'R$^2$='+format(rr22[idx1],'.2f')+'\nRMSE='+ format(rmse22[idx1],'.2f') + ' K\nMAE=' + format(mae22[idx1],'.2f') + ' K'
    plt.text(0.04, 0.86, txt1, fontsize=9,horizontalalignment='left',verticalalignment='top',transform=ax.transAxes)
    
    plt.text(0.98, 0.205, "Clear-Sky LST", fontsize=8, horizontalalignment='right',verticalalignment='bottom',transform=ax.transAxes)
    txt2 = 'R$^2$='+format(rr21[idx1],'.2f')+'\nRMSE='+ format(rmse21[idx1],'.2f') + ' K\nMAE=' + format(mae21[idx1],'.2f') + ' K'
    plt.text(0.98, 0.02, txt2, fontsize=9,horizontalalignment='right',verticalalignment='bottom',transform=ax.transAxes)
    ax = plt.gca()
    ax.add_artist(legend1)
    plt.text(0.1, 0.025, codes[idx1], fontsize=8, horizontalalignment='left',verticalalignment='bottom',transform=ax.transAxes)
    
    
    plt.xlim(x1lim,x2lim)
    plt.ylim(x1lim,x2lim)
    plt.ylabel(ylabel)
    
    
    ### 4
    plt.subplot(324)
    idx1 = 3
    plt.plot([-50,1000],[-50,1000],color='black',lw=1)
    ax = plt.gca()
    sc1 = ax.scatter(data22[idx1][0],data22[idx1][1],s=60,facecolors='none',color='blue',marker='o',label=' ')
    legend1 = ax.legend(handles=[sc1], loc=2,fontsize=8,frameon=False,bbox_to_anchor=(-0.03, 1.00))
    sc2 = plt.scatter(data21[idx1][0],data21[idx1][1],s=60,facecolors='none',color='red',marker='v',label=' ')
    legend2 = plt.legend(handles=[sc2],loc=4,fontsize=8,frameon=False,bbox_to_anchor=(0.77, 0.165))
    plt.text(0.11, 0.98, "Cloud-Covered or\nNo Landsat Overpass", fontsize=8,horizontalalignment='left',verticalalignment='top',transform=ax.transAxes)
    txt1 = 'R$^2$='+format(rr22[idx1],'.2f')+'\nRMSE='+ format(rmse22[idx1],'.2f') + ' K\nMAE=' + format(mae22[idx1],'.2f') + ' K'
    plt.text(0.04, 0.86, txt1, fontsize=9,horizontalalignment='left',verticalalignment='top',transform=ax.transAxes)
    
    plt.text(0.98, 0.205, "Clear-Sky LST", fontsize=8, horizontalalignment='right',verticalalignment='bottom',transform=ax.transAxes)
    txt2 = 'R$^2$='+format(rr21[idx1],'.2f')+'\nRMSE='+ format(rmse21[idx1],'.2f') + ' K\nMAE=' + format(mae21[idx1],'.2f') + ' K'
    plt.text(0.98, 0.02, txt2, fontsize=9,horizontalalignment='right',verticalalignment='bottom',transform=ax.transAxes)
    ax = plt.gca()
    ax.add_artist(legend1)
    plt.text(0.1, 0.025, codes[idx1], fontsize=8, horizontalalignment='left',verticalalignment='bottom',transform=ax.transAxes)
    
    plt.xlim(x1lim,x2lim)
    plt.ylim(x1lim,x2lim)
    
    
    
    ### sub5
    plt.subplot(325)
    idx1 = 4
    plt.plot([-50,1000],[-50,1000],color='black',lw=1)
    ax = plt.gca()
    sc1 = ax.scatter(data22[idx1][0],data22[idx1][1],s=60,facecolors='none',color='blue',marker='o',label=' ')
    legend1 = ax.legend(handles=[sc1], loc=2,fontsize=8,frameon=False,bbox_to_anchor=(-0.03, 1.00))
    sc2 = plt.scatter(data21[idx1][0],data21[idx1][1],s=60,facecolors='none',color='red',marker='v',label=' ')
    legend2 = plt.legend(handles=[sc2],loc=4,fontsize=8,frameon=False,bbox_to_anchor=(0.77, 0.165))
    plt.text(0.11, 0.98, "Cloud-Covered or\nNo Landsat Overpass", fontsize=8,horizontalalignment='left',verticalalignment='top',transform=ax.transAxes)
    txt1 = 'R$^2$='+format(rr22[idx1],'.2f')+'\nRMSE='+ format(rmse22[idx1],'.2f') + ' K\nMAE=' + format(mae22[idx1],'.2f') + ' K'
    plt.text(0.04, 0.86, txt1, fontsize=9,horizontalalignment='left',verticalalignment='top',transform=ax.transAxes)
    
    plt.text(0.98, 0.205, "Clear-Sky LST", fontsize=8, horizontalalignment='right',verticalalignment='bottom',transform=ax.transAxes)
    txt2 = 'R$^2$='+format(rr21[idx1],'.2f')+'\nRMSE='+ format(rmse21[idx1],'.2f') + ' K\nMAE=' + format(mae21[idx1],'.2f') + ' K'
    plt.text(0.98, 0.02, txt2, fontsize=9,horizontalalignment='right',verticalalignment='bottom',transform=ax.transAxes)
    ax = plt.gca()
    ax.add_artist(legend1)
    plt.text(0.1, 0.025, codes[idx1], fontsize=8, horizontalalignment='left',verticalalignment='bottom',transform=ax.transAxes)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(x1lim,x2lim)
    plt.ylim(x1lim,x2lim)
    
    
    ### sub6
    plt.subplot(326)
    
    idx1 = 5
    plt.plot([-50,1000],[-50,1000],color='black',lw=1)
    ax = plt.gca()
    sc1 = ax.scatter(data22[idx1][0],data22[idx1][1],s=60,facecolors='none',color='blue',marker='o',label=' ')
    legend1 = ax.legend(handles=[sc1], loc=2,fontsize=8,frameon=False,bbox_to_anchor=(-0.03, 1.00))
    sc2 = plt.scatter(data21[idx1][0],data21[idx1][1],s=60,facecolors='none',color='red',marker='v',label=' ')
    legend2 = plt.legend(handles=[sc2],loc=4,fontsize=8,frameon=False,bbox_to_anchor=(0.77, 0.165))
    plt.text(0.11, 0.98, "Cloud-Covered or\nNo Landsat Overpass", fontsize=8,horizontalalignment='left',verticalalignment='top',transform=ax.transAxes)
    txt1 = 'R$^2$='+format(rr22[idx1],'.2f')+'\nRMSE='+ format(rmse22[idx1],'.2f') + ' K\nMAE=' + format(mae22[idx1],'.2f') + ' K'
    plt.text(0.04, 0.86, txt1, fontsize=9,horizontalalignment='left',verticalalignment='top',transform=ax.transAxes)
    
    plt.text(0.98, 0.205, "Clear-Sky LST", fontsize=8, horizontalalignment='right',verticalalignment='bottom',transform=ax.transAxes)
    txt2 = 'R$^2$='+format(rr21[idx1],'.2f')+'\nRMSE='+ format(rmse21[idx1],'.2f') + ' K\nMAE=' + format(mae21[idx1],'.2f') + ' K'
    plt.text(0.98, 0.02, txt2, fontsize=9,horizontalalignment='right',verticalalignment='bottom',transform=ax.transAxes)
    ax = plt.gca()
    ax.add_artist(legend1)
    plt.text(0.1, 0.025, codes[idx1], fontsize=8, horizontalalignment='left',verticalalignment='bottom',transform=ax.transAxes)
    plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    plt.xlim(x1lim,x2lim)
    plt.ylim(x1lim,x2lim)
    
    plt.tight_layout()
    # plt.savefig('fig/ldn.pdf')
    # plt.savefig('fig/ldn.png')
    # plt.savefig('fig/ldn.jpg')
    plt.savefig('fig/resultLDN.pdf')
    plt.savefig('fig/resultLDN.png')
    plt.savefig('fig/resultLDN.jpg')
    plt.show()  




















