'''
This script algorithmically identifies heatwaves based on the definition of Russo et al., 2015.
First, daily maximum temperatures are compared to the daily Q90 climatology.
In Russo et al., a heatwave is a period of 3+ days with T > Q90.
We added another condition: at least {SPATIAL_EXTENT}% of the region gridpoints must 
exceed this threshold at the same time.
Note: checking for sequences diminishes the heatwave area, checking for sufficient area can split sequences.
Therefore, these two operations are done in a loop until the number of identified heatwave days stays the same.
That is: all heatwave days were found that are a part of a sufficiently large and sufficiently long heat period. 
'''
import numpy as np 
import xarray as xr
from global_land_mask import globe
from scipy.signal import convolve2d

from config import PATH_DATA_HW, REGION, REGION_NAME, SPATIAL_EXTENT

LATITUDE_NORTH, LATITUDE_SOUTH, LONGITUDE_WEST, LONGITUDE_EAST = REGION[REGION_NAME]


ds_Tmax = xr.open_dataset(PATH_DATA_HW + 'ds_Tmax_daily.nc')
Q90 = xr.open_dataarray(PATH_DATA_HW + 'ds_q90.nc')

#these two dataarrays are only used when determining the HWMID value of a heatwave
Q25 = xr.open_dataarray(PATH_DATA_HW + 'ds_q25.nc')
Q75 = xr.open_dataarray(PATH_DATA_HW + 'ds_q75.nc')

#drop leap days so that each year has 365 days
#otherwise the comparison is immposible - no climatology for Feb 29
ds_Tmax = ds_Tmax.sel(time=~((ds_Tmax.time.dt.month == 2) & (ds_Tmax.time.dt.day == 29))) 
ds_Tmax = ds_Tmax.sel(lat=slice(LATITUDE_NORTH, LATITUDE_SOUTH), lon=slice(LONGITUDE_WEST, LONGITUDE_EAST))

def compare_temps(ds_T, q90):
    rs_years = ds_T.resample(time='1Y')
    y = []

    for y_string in rs_years.groups.values():
        year = ds_T.isel(time=y_string)
        nr_days = len(year['time'])
        
        #do this because of a year that can be incomplete, <365 days
        q = q90.isel(doy=slice(0, nr_days)) 
        #assign this years date to Q90 so that it can be compared
        q['doy'] = year['time'].to_numpy()
        q = q.rename({'doy':'time'})
        compare = year > q
        y.append(compare)

    ds_T_overQ90 = xr.concat(y, dim='time')
    ds_T_overQ90 = ds_T_overQ90.rename({'T':'bool_T'})

    return ds_T_overQ90

def spatial_extent_func(grid, nr_land_gridpoints):
    #number of land gridpoint where T exceeded Q90
    nr_overQ90 = np.sum(grid)
    percentage_over_Q90 = (nr_overQ90 / nr_land_gridpoints) * 100

    #if on a given day, the spatial extent of the hot days exceeds the threshold
    #this day can constitute a part of a heatwave
    #else change all True values to False, and this day will not be 
    #recognized as a potential heatwave day
    if percentage_over_Q90 >= SPATIAL_EXTENT:
        return grid
    else:
        #where True, change to False
        #copy, because the original grid passed by apply_ufunc is immutable
        g = grid.copy()
        g[grid] = False
        return g

#day by day, check if enough land grid points are under heatwave conditions
def spatial_extent(ds):
    #ds contains only False and True values
    #some of the False values - temperature was not high enough
    #the rest of False values - temperature was NaN thanks to ocean masking
    lat, lon = ds['lat'].values, ds['lon'].values
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    land_mask = globe.is_land(lat_grid, lon_grid)
    #when checking the percentage of gridpoints exceeding Q90, we only consider land
    nr_land_gridpoints = np.sum(land_mask)

    ds_spatial = xr.apply_ufunc(
        spatial_extent_func,
        ds['bool_T'],
        input_core_dims=[['lat', 'lon']],
        output_core_dims=[['lat', 'lon']],
        vectorize=True,
        kwargs={'nr_land_gridpoints':nr_land_gridpoints}
    )
    return ds_spatial.to_dataset()

def find_heatwaves_func(timeseries):
    #find the indices of True values
    true_indices = np.where(timeseries)
    #find consecutive sequences of True values, no matter their length
    sequences = np.split(true_indices[0], np.where(np.diff(true_indices[0]) != 1)[0] + 1)
    #sequences with length three or more
    long_sequences = [seq for seq in sequences if len(seq) >= 3]

    #boolean array with the identified sequences
    result = np.zeros_like(timeseries, dtype=bool)
    for seq in long_sequences:
        result[seq] = True

    return result

#function used to finding sequences of three and more consecutive heatwave days
def find_heatwaves(ds):
    hw = xr.apply_ufunc(
        find_heatwaves_func,
        ds['bool_T'],
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True
    )
    return hw.to_dataset()

#select heatwave days latitude by latitude
#returns only unique values
def heatwave_days(ds_HW):    
    hw_dates = []
    for l in ds_HW.lat:
        ds_l = ds_HW.sel(lat=l)
        time_hw_l = ds_l.time.where(ds_l.bool_T)
        time_np = time_hw_l.to_numpy().flatten()
        hw_dates.append(np.unique(time_np)[:-1])

    flatten_hw_dates = [d for dlat in hw_dates for d in dlat]
    dates_unique = np.array(flatten_hw_dates)
    dates_unique = np.unique(flatten_hw_dates)

    return dates_unique

#using all previous functions, find all heatwaves according
#to the definition. Then return all heatwave days.
def identify_HW_days(ds_T, Q90):
    ds_T_overQ90 = compare_temps(ds_T, Q90)
    ds_spatial = spatial_extent(ds_T_overQ90)
    ds_HW = find_heatwaves(ds_spatial)

    hw_days_prev = len(heatwave_days(ds_HW))
    while True:
        ds_HW = spatial_extent(ds_HW)
        ds_HW = find_heatwaves(ds_HW)
        hw_days = heatwave_days(ds_HW)
        nr_hw_days = len(hw_days)
        if hw_days_prev - nr_hw_days == 0:
            break
        else:
            hw_days_prev = nr_hw_days
    
    return hw_days


#####################################################

def convolve(grid, filter, convolve_landmask):
    convolution = convolve2d(grid, filter)
    #max value in convolution is 50
    thr = 0.9 * filter.shape[0] * filter.shape[1]
    #spatial_threshold = 0.90 * convolve_landmask
    #this means that at least 90 percent of the land area was under heatwave conditions
    #compare_land = convolution > spatial_threshold #boolean array
    maximum = convolution.max()

    #if maximum >= spatial_threshold:
    if maximum >= thr:
        return grid
    else:
        #where True, change to False
        #copy, because the original grid passed by apply_ufunc is immutable
        g = grid.copy()
        g[grid] = False
        return g

def spatial_extent_conv(ds):
    filter = np.ones((10,10))

    lat, lon = ds['lat'].values, ds['lon'].values
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    land_mask = globe.is_land(lat_grid, lon_grid)
    convolve_landmask = convolve2d(land_mask, filter)

    ds_spatial = xr.apply_ufunc(
        convolve,
        ds['bool_T'],
        input_core_dims=[['lat', 'lon']],
        output_core_dims=[['lat', 'lon']],
        vectorize=True,
        kwargs={'filter':filter, 'convolve_landmask':convolve_landmask}
        
    )
    return ds_spatial.to_dataset()



def identify_HW_days_conv(ds_T, Q90):
    ds_T_overQ90 = compare_temps(ds_T, Q90)
    ds_spatial = spatial_extent_conv(ds_T_overQ90)
    ds_HW = find_heatwaves(ds_spatial)

    hw_days_prev = len(heatwave_days(ds_HW))
    while True:
        ds_HW = spatial_extent_conv(ds_HW)
        ds_HW = find_heatwaves(ds_HW)
        hw_days = heatwave_days(ds_HW)
        nr_hw_days = len(hw_days)
        if hw_days_prev - nr_hw_days == 0:
            break
        else:
            hw_days_prev = nr_hw_days
    
    return hw_days



'''
this works: SPATIAL_EXTENT set to 20% gives 582, 581, 566 (north to south, Europe) heatwave days during 
            summers 1979 - 2022, with the year 2010 approximately in the middle of the day list. 
            The same calculation gives a very similar number of heatwave days if we consider the
            whole year, not just the summer season ... around 1330 - 1335. 
'''

