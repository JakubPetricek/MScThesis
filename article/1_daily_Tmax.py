'''
This script calculates the threshold (Q90) used to identify heatwaves, defined according to
Russo et al., 2015, and generates the other data (ds_Tmax, Q25, Q75) neccessary to determining
the heatwave magnitude index daily (HWMId)

The calculation of the HWMID uses maximal daily temperatures.
In this script, we load the input data -- 6 hourly temperature reanalysis, and from each day, 
we select the maximum temperature. This dataset with one value per day is then saved.
'''

import numpy as np
import pandas as pd
import xarray as xr
import os
from global_land_mask import globe

from config import PATH_GRB_T, PATH_DATA_HW, REGION, REGION_NAME, START_CLIMATOLOGY, END_CLIMATOLOGY

LATITUDE_NORTH, LATITUDE_SOUTH, LONGITUDE_WEST, LONGITUDE_EAST = REGION[REGION_NAME]

ds = xr.open_mfdataset(PATH_GRB_T + '*.grb', parallel=True, combine='nested', concat_dim='time', engine='cfgrib')
ds = ds.drop_vars(['number', 'step', 'surface', 'valid_time'])
ds = ds.rename({'latitude':'lat', 'longitude': 'lon', 't2m':'T'})
ds['T'] -= 273.15
ds.T.attrs = {'long_name': '2m temperature', 'units': '°C'}
ds.attrs = {'Description' : 'Daily Tmax over land'}

#select the studied region
ds_region = ds.sel(lat=slice(LATITUDE_NORTH, LATITUDE_SOUTH), lon=slice(LONGITUDE_WEST, LONGITUDE_EAST))

#from each day and at each gridpoint, select the maximum temperature
ds_Tmax = ds_region.resample(time='1D').max(dim='time')

#now save the data and load the smaller dataset again, it speeds up the calculation of Q90
if not os.path.exists(PATH_DATA_HW):
    os.makedirs(PATH_DATA_HW)

#select only values over land, not over the ocean
lon_grid, lat_grid = np.meshgrid(ds_region.lon.values, ds_region.lat.values)
land_mask = globe.is_land(lat_grid, lon_grid)

ds_Tmax_land = ds_Tmax.where(land_mask)
ds_Tmax_land.to_netcdf(PATH_DATA_HW + f'ds_Tmax_daily.nc')

#load the data again
ds_Tmax_land = xr.open_dataset(PATH_DATA_HW + f'ds_Tmax_daily.nc')

#now, calculation of Q90, the 90th percentile of a dataset A_d defined in the paper
#Top ten European heatwaves since 1950 and their occurrence in the coming decades, Russo et al., 2015
#Q90 is the climatology used for determining heatwaves
#for day of the year d, select a 31-day window centered around d ... (d-15, d+15)
#calculate the 90th percentile of this dataset - the value of Q90 for the given day

def Q90(ds, start_climatology, end_climatology):
    #random year, just to get the date range 01-01 to 12-31 to use later
    days_in_year = pd.date_range(start='1979-01-01', end='1979-12-31')
    #what years are considered for the climatology calculation
    years_climatology = np.arange(start_climatology, end_climatology+1)
    Q90 = []
    for d in days_in_year:
        mmdd = d.strftime('%m-%d')
        delta = pd.to_timedelta('15 days')
        #31-day window centered around d, selected for each year
        A_d_dates = np.array([pd.date_range(start=pd.to_datetime(str(y)+'-'+mmdd)-delta,
                            end=pd.to_datetime(str(y)+'-'+mmdd)+delta) 
                            for y in years_climatology]).flatten()
        
        #a special case for the first 15 days of Jan 1979 
        day_int = int(d.strftime('%d'))
        if d.strftime('%m') == '01' and day_int <= 15:
            A_d_dates = A_d_dates[(16-day_int):]

        #select data 
        A_d = ds.sel(time=(ds.time.isin(A_d_dates)))
        #when calculating quantiles, the whole array must be loaded into memory at once
        #therefore we need to rechunk the array
        Q90.append(A_d.chunk({"time": -1}).quantile(q=0.9, dim='time'))
    
    #number the resulting (lat, lon) grids by the day of the year - doy
    for ix, _ in enumerate(days_in_year):
        Q90[ix] = Q90[ix].assign_coords({'doy': ix+1})
        Q90[ix] = Q90[ix].expand_dims(dim={'doy': 1})

    Q90 = xr.combine_by_coords(Q90) # (lat, lon, doy)
    Q90 = Q90.drop(['quantile'])
    Q90.attrs = {'Description' : 'Heatwave climatology - 90th percentile of a selected temperature dataset'}
    Q90.T.attrs = {'long_name':'temperature', 'short_name':'T', 'units':'°C'} 

    return Q90

#general quantile calculation, will be used for calculating Q25 and Q75
def q(ds, percentile):
    qq = ds.sel(time=slice(f'{START_CLIMATOLOGY}', f'{END_CLIMATOLOGY}')).chunk(dict(time=-1)).load().resample(time='1Y').max(dim='time').quantile(q=percentile, dim='time')
    qq = qq.drop(['quantile'])
    qq.T.attrs = {'long_name':'temperature', 'short_name':'T', 'units':'°C'}

    return qq

#########
#Run the calculations
#########

#time series of length 365, shape (365, lat, lon)
q90_land = Q90(ds_Tmax_land, START_CLIMATOLOGY, END_CLIMATOLOGY)

#the 25th and 75th percentile for every gridpoint, shape (lat, lon)
q25_land = q(ds_Tmax_land, 0.25)
q75_land = q(ds_Tmax_land, 0.75)

#save data, the folder already exists
q90_land.to_netcdf(PATH_DATA_HW + f'ds_q90.nc')
q25_land.to_netcdf(PATH_DATA_HW + f'ds_q25.nc')
q75_land.to_netcdf(PATH_DATA_HW + f'ds_q75.nc')