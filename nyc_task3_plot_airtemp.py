# -*- coding: utf-8 -*-
"""
This is the NYC air temperature data plotting script.
    Reproduce Figure 12. 

@author: skrisliu

Go to skrisliu.com/delag for updates. 

NYC
72502, Newark Airport, 40.68275°, -74.16927°

KJRB0                 New York / Wall Street 
72502                         Newark Airport
KNYC0              New York City / Yorkville
KTEB0                              Teterboro
74486                John F. Kennedy Airport
KLDJ0                                 Linden
KCDW0                   Caldwell / Fairfield
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
#import pvlib 
from pvlib.location import Location
from scipy.stats import pearsonr



site = 'nyc'
year = '2023'

PRED_PATH = 'nyc/nyc_pred_2023.npz'   # download from release

BIGX = []
BIGY = []
BIGMASK = []


#%% Get Stations
stations = Stations()
stations = stations.nearby(40.7, -74.0)
station = stations.fetch(50)
if False:
    station.to_csv(site+'/'+site+'_stations.csv')
    
#%%
codes = ['KJRB0', '72502', 'KNYC0', 'KTEB0', '74486', 'KLDJ0', 'KCDW0']

start = datetime(2023, 1, 1)
end = datetime(2023, 12, 31)



#%% load dem ndvi
demdata = np.load('nyc/nyc_dem_2023.npy')
dems = []


ndvi = np.load('nyc/nyc_bands_2023.npy')
ndvi = np.float32(ndvi)
ndvi = (ndvi[3,:,:] - ndvi[2,:,:])  / (ndvi[3,:,:] + ndvi[2,:,:])
ndvis = []



#%%
# 0-570-1140-1733
# 0-560-1120-1667
im0 = np.zeros([365,1733,1667],dtype=np.float32)


im0 = np.load(PRED_PATH)['a']
im0 = im0.astype(np.float32)
im0 = im0*0.00341802 + 149.0


#%%
fig = plt.figure(figsize=(8,6),dpi=200)
plt.imshow(im0[290,:,:],vmin=280,vmax=310)
plt.colorbar()
plt.tight_layout()
plt.show()


#%% geo prj
site = 'nyc'

im = gdal.Open(site + '/t20230221_SR_B1.TIF', gdal.GA_ReadOnly)
geo = im.GetGeoTransform()
prj = im.GetProjection()





#%% Get UTM, station KJRB0
code = 'KJRB0'
df = pd.read_csv(site+'/'+site+'_stations.csv')
stat = df[df['id']==code]
lat = stat['latitude'].values[0]
lon = stat['longitude'].values[0]

utmx = utm.from_latlon(latitude=lat, longitude=lon, force_zone_number=18)
x0 = int((utmx[0]-geo[0]) / geo[1])
y0 = int((utmx[1]-geo[3]) / geo[5])
y_pre0 = im0[:,y0,x0]

df_gt = pd.read_pickle(site+'/'+site+'2023_'+code+'.pkl')
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
reg.intercept_     
y_pre4 = reg.predict(x)

np.mean(y_gt) - np.mean(y_pre4)         
mean_absolute_error(y_gt, y_pre4)        
root_mean_squared_error(y_gt, y_pre4)  


BIGX.append(x)
BIGY.append(y)

### clear date
fp = site+'/datacube/'+site+'2023clearmasks.npy'
clear = np.load(fp)
y_clear = clear[:,y0,x0]


BIGMASK.append(y_clear)


### dem data
dems.append(np.ones([365,])*demdata[y0,x0])
ndvis.append(np.ones([365,]) *ndvi[y0,x0] )


#%%


#%% Station 2. Get UTM, station 72502
### Newark Airport

scode = '72502'
df = pd.read_csv(site+'/'+site+'_stations.csv')
stat = df[df['id']==scode]
lat = stat['latitude'].values[0]
lon = stat['longitude'].values[0]
lat =  40.68275
lon =  -74.16927

utmx = utm.from_latlon(latitude=lat, longitude=lon, force_zone_number=18)
x0 = int((utmx[0]-geo[0]) / geo[1])
y0 = int((utmx[1]-geo[3]) / geo[5])
y_pre0 = im0[:,y0,x0]

df_gt = pd.read_pickle(site+'/'+site+'2023_'+scode+'.pkl')
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
reg.intercept_    
y_pre4 = reg.predict(x)

np.mean(y_gt) - np.mean(y_pre4)         
mean_absolute_error(y_gt, y_pre4)        
root_mean_squared_error(y_gt, y_pre4)  


BIGX.append(x)
BIGY.append(y)


### clear date
fp = site+'/datacube/'+site+'2023clearmasks.npy'
clear = np.load(fp)
y_clear = clear[:,y0,x0]

BIGMASK.append(y_clear)

### dem data
dems.append(np.ones([365,])*demdata[y0,x0])
ndvis.append(np.ones([365,]) *ndvi[y0,x0] )



#%% Station 3

#%% Station 3. Get UTM, station KNYC0
scode = 'KNYC0'
df = pd.read_csv(site+'/'+site+'_stations.csv')
stat = df[df['id']==scode]
lat = stat['latitude'].values[0]
lon = stat['longitude'].values[0]

utmx = utm.from_latlon(latitude=lat, longitude=lon, force_zone_number=18)
x0 = int((utmx[0]-geo[0]) / geo[1])
y0 = int((utmx[1]-geo[3]) / geo[5])
y_pre0 = im0[:,y0,x0]

df_gt = pd.read_pickle(site+'/'+site+'2023_'+scode+'.pkl')
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
reg.intercept_    
y_pre4 = reg.predict(x)

np.mean(y_gt) - np.mean(y_pre4)         
mean_absolute_error(y_gt, y_pre4)        
root_mean_squared_error(y_gt, y_pre4)  


BIGX.append(x)
BIGY.append(y)


### clear date
fp = site+'/datacube/'+site+'2023clearmasks.npy'
clear = np.load(fp)
y_clear = clear[:,y0,x0]

BIGMASK.append(y_clear)

### dem data
dems.append(np.ones([365,])*demdata[y0,x0])
ndvis.append(np.ones([365,]) *ndvi[y0,x0] )



#%% Station 4


#%% Station 4. Get UTM, station KTEB0
scode = 'KTEB0'
df = pd.read_csv(site+'/'+site+'_stations.csv')
stat = df[df['id']==scode]
lat = stat['latitude'].values[0]
lon = stat['longitude'].values[0]

utmx = utm.from_latlon(latitude=lat, longitude=lon, force_zone_number=18)
x0 = int((utmx[0]-geo[0]) / geo[1])
y0 = int((utmx[1]-geo[3]) / geo[5])
y_pre0 = im0[:,y0,x0]

df_gt = pd.read_pickle(site+'/'+site+'2023_'+scode+'.pkl')
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
reg.intercept_    
y_pre4 = reg.predict(x)

np.mean(y_gt) - np.mean(y_pre4)         
mean_absolute_error(y_gt, y_pre4)        
root_mean_squared_error(y_gt, y_pre4)  


BIGX.append(x)
BIGY.append(y)


### clear date
fp = site+'/datacube/'+site+'2023clearmasks.npy'
clear = np.load(fp)
y_clear = clear[:,y0,x0]

BIGMASK.append(y_clear)


### dem data
dems.append(np.ones([365,])*demdata[y0,x0])
ndvis.append(np.ones([365,]) *ndvi[y0,x0] )



#%% Station 5


#%% Station 5. Get UTM, station 74486
scode = '74486'
df = pd.read_csv(site+'/'+site+'_stations.csv')
stat = df[df['id']==scode]
lat = stat['latitude'].values[0]
lon = stat['longitude'].values[0]

utmx = utm.from_latlon(latitude=lat, longitude=lon, force_zone_number=18)
x0 = int((utmx[0]-geo[0]) / geo[1])
y0 = int((utmx[1]-geo[3]) / geo[5])
y_pre0 = im0[:,y0,x0]

df_gt = pd.read_pickle(site+'/'+site+'2023_'+scode+'.pkl')
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
reg.intercept_   
y_pre4 = reg.predict(x)

np.mean(y_gt) - np.mean(y_pre4)         
mean_absolute_error(y_gt, y_pre4)        
root_mean_squared_error(y_gt, y_pre4)  


BIGX.append(x)
BIGY.append(y)


### clear date
fp = site+'/datacube/'+site+'2023clearmasks.npy'
clear = np.load(fp)
y_clear = clear[:,y0,x0]

BIGMASK.append(y_clear)

### dem data
dems.append(np.ones([365,])*demdata[y0,x0])
ndvis.append(np.ones([365,]) *ndvi[y0,x0] )





#%% Station 6

#%% Station 6. Get UTM, station KLDJ0
scode = 'KLDJ0'
df = pd.read_csv(site+'/'+site+'_stations.csv')
stat = df[df['id']==scode]
lat = stat['latitude'].values[0]
lon = stat['longitude'].values[0]

utmx = utm.from_latlon(latitude=lat, longitude=lon, force_zone_number=18)
x0 = int((utmx[0]-geo[0]) / geo[1])
y0 = int((utmx[1]-geo[3]) / geo[5])
y_pre0 = im0[:,y0,x0]

df_gt = pd.read_pickle(site+'/'+site+'2023_'+scode+'.pkl')
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
reg.intercept_    
y_pre4 = reg.predict(x)

np.mean(y_gt) - np.mean(y_pre4)         
mean_absolute_error(y_gt, y_pre4)        
root_mean_squared_error(y_gt, y_pre4)  


BIGX.append(x)
BIGY.append(y)


### clear date
fp = site+'/datacube/'+site+'2023clearmasks.npy'
clear = np.load(fp)
y_clear = clear[:,y0,x0]

BIGMASK.append(y_clear)


### dem data
dems.append(np.ones([365,])*demdata[y0,x0])
ndvis.append(np.ones([365,]) *ndvi[y0,x0] )






#%% Station 7


#%% Station 7. Get UTM, station KCDW0
scode = 'KCDW0'
df = pd.read_csv(site+'/'+site+'_stations.csv')
stat = df[df['id']==scode]
lat = stat['latitude'].values[0]
lon = stat['longitude'].values[0]

utmx = utm.from_latlon(latitude=lat, longitude=lon, force_zone_number=18)
x0 = int((utmx[0]-geo[0]) / geo[1])
y0 = int((utmx[1]-geo[3]) / geo[5])
y_pre0 = im0[:,y0,x0]

df_gt = pd.read_pickle(site+'/'+site+'2023_'+scode+'.pkl')
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
reg.intercept_    
y_pre4 = reg.predict(x)

np.mean(y_gt) - np.mean(y_pre4)         
mean_absolute_error(y_gt, y_pre4)        
root_mean_squared_error(y_gt, y_pre4)  


BIGX.append(x)
BIGY.append(y)


### clear date
fp = site+'/datacube/'+site+'2023clearmasks.npy'
clear = np.load(fp)
y_clear = clear[:,y0,x0]

BIGMASK.append(y_clear)


### dem data
dems.append(np.ones([365,])*demdata[y0,x0])
ndvis.append(np.ones([365,]) *ndvi[y0,x0] )





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




#%% seperate fit
### clear
x2a = BIGX2[BIGMASK2]
y2a = BIGY2[BIGMASK2]

reg1 = LinearRegression().fit(x2a,y2a)
reg1.coef_
reg1.intercept_
ypre2a = reg1.predict(x2a)

np.mean(y2a) - np.mean(ypre2a)         
mean_absolute_error(y2a, ypre2a)        
root_mean_squared_error(y2a, ypre2a) 
pearsonr(y2a, ypre2a)


### cloudy/no obs
x2b = BIGX2[~BIGMASK2]
y2b = BIGY2[~BIGMASK2]

reg2 = LinearRegression().fit(x2b,y2b)
reg2.coef_
reg2.intercept_
ypre2b = reg2.predict(x2b)

np.mean(y2b) - np.mean(ypre2b)         
mean_absolute_error(y2b, ypre2b)        
root_mean_squared_error(y2b, ypre2b) 
pearsonr(y2b, ypre2b)

y11 = reg1.predict(BIGX2) # clear
y12 = reg2.predict(BIGX2) # cloudy


#%% get data
data21 = [] # clear
data22 = [] # cloudcover

bias22 = []
mae22 = []
rmse22 = []
rr22 = []

bias21 = []
mae21 = []
rmse21 = []
rr21 = []



for n11 in np.arange(0,2555,365):
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
codes = ['KJRB', 'KEWR', 'KNYC', 'KTEB', 'KJFK', 'KLDJ','KCDW']




if True:
    xlabel = 'Observed T$_{air}$ (K)'
    ylabel = 'Estimated T$_{air}$ (K)'
    x1lim,x2lim = 260,310
    n2 = np.sum(BIGMASK2)
    n1 = BIGMASK2.shape[0] - n2
    fig = plt.figure(figsize=(7,12), dpi=300)
    
    ###### sub 2
    plt.subplot(421)
    idx1 = 0
    plt.plot([-50,1000],[-50,1000],color='black',lw=1)
    ax = plt.gca()
    sc1 = ax.scatter(data22[idx1][0],data22[idx1][1],s=60,facecolors='none',color='blue',marker='o',label=' ')
    legend1 = ax.legend(handles=[sc1], loc=2,fontsize=8,frameon=False,bbox_to_anchor=(-0.03, 1.00))
    sc2 = plt.scatter(data21[idx1][0],data21[idx1][1],s=60,facecolors='none',color='red',marker='v',label=' ')
    legend2 = plt.legend(handles=[sc2],loc=4,fontsize=8,frameon=False,bbox_to_anchor=(0.77, 0.165))
    # ax = plt.gca()
    plt.text(0.11, 0.98, "Cloud-Covered or\nNo Landsat Overpass", fontsize=8,horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
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
    
    
    plt.subplot(423)
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
    plt.ylabel(ylabel)

    

    ###### sub 2
    plt.subplot(424)
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
    
    
    
    
    
    
    ### sub 3
    plt.subplot(425)
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
    plt.ylabel(ylabel)
    
    
    ### 4
    plt.subplot(426)
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
    
    plt.xlim(x1lim,x2lim)
    plt.ylim(x1lim,x2lim)
    
    
    
    ### sub5
    plt.subplot(427)
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
    plt.ylabel(ylabel)
    plt.xlim(x1lim,x2lim)
    plt.ylim(x1lim,x2lim)
    
    
    ### sub6
    plt.subplot(428)
    
    idx1 = 6
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
    plt.savefig('fig/resultNYC.pdf')
    plt.savefig('fig/resultNYC.png')
    plt.savefig('fig/resultNYC.jpg')
    plt.show()  


























