import xarray as xr
import numpy as np
import os as os
from identify_heatwaves import identify_HW_days, identify_HW_days_conv

from config import PATH_DATA, PATH_DATA_HW, PRESS_LEVEL, N, LAT_BANDS, TIME_WINDOW
CONVOLUTION = False

#drop leap days
ds_ptp = xr.open_dataset(PATH_DATA + f'ds_ptp_Z{PRESS_LEVEL}_GP1-{N}.nc')
ds_ptp = ds_ptp.sel(time=~((ds_ptp.time.dt.month == 2) & (ds_ptp.time.dt.day == 29)))

ds_speed = xr.open_dataset(PATH_DATA + f'ds_speed_Z{PRESS_LEVEL}_GP1-{N}_{TIME_WINDOW}_days.nc')
ds_speed = ds_speed.sel(time=~((ds_speed.time.dt.month == 2) & (ds_speed.time.dt.day == 29)))

#ptp, speed - data daily at 9am... Tmax ... at midnight
#does not really work as intended, but makes the rest easier 
ds_ptp['time'] = ds_ptp['time'].astype('datetime64[D]') 
ds_speed['time'] = ds_speed['time'].astype('datetime64[D]') 

#select ds_speed time, because that is the shortest time coordinate from all ds
ds_Tmax = xr.open_dataset(PATH_DATA_HW + 'ds_Tmax_daily.nc').sel(time=ds_speed.time)
Q90 = xr.open_dataarray(PATH_DATA_HW + 'ds_q90.nc')


def ds_during_HW(ds, bands, ds_T, Q90):
    ds_HW = []
    ds_outside_HW = []
    for band in bands:
        ds_band = ds.sel(lat=band)
        #find heatwave days in given latitude band
        hw_days_band = identify_HW_days(ds_T.sel(lat=band), Q90.sel(lat=band))
        #heatwave data
        ds_HW.append(ds_band.sel(time=hw_days_band))
        #non-heatwave data
        ds_outside_HW.append(ds_band.sel(time=~np.isin(ds_band.time, hw_days_band)))

    
    ds_HW = xr.concat(ds_HW, 'lat').drop_duplicates(dim='lat')
    ds_outside_HW = xr.concat(ds_outside_HW, 'lat').drop_duplicates(dim='lat')

    return ds_HW, ds_outside_HW
    

ds_ptp_HW, ds_ptp_outside_HW = ds_during_HW(ds_ptp, LAT_BANDS, ds_Tmax, Q90)
ds_speed_HW, ds_speed_outside_HW = ds_during_HW(ds_speed, LAT_BANDS, ds_Tmax, Q90)

if not os.path.exists(PATH_DATA):
    os.makedirs(PATH_DATA)

ds_speed_HW.to_netcdf(PATH_DATA + f'ds_HW_speed_Z{PRESS_LEVEL}_GP1-{N}_{TIME_WINDOW}_days.nc')
ds_speed_outside_HW.to_netcdf(PATH_DATA + f'ds_outside_HW_speed_Z{PRESS_LEVEL}_GP1-{N}_{TIME_WINDOW}_days.nc')

ds_ptp_HW.to_netcdf(PATH_DATA + f'ds_HW_ptp_Z{PRESS_LEVEL}_GP1-{N}.nc')
ds_ptp_outside_HW.to_netcdf(PATH_DATA + f'ds_outside_HW_ptp_Z{PRESS_LEVEL}_GP1-{N}.nc')



