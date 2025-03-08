import xarray as xr
import numpy as np
import os
from scipy.signal import find_peaks
from auxiliary_scripts import extend_ds

from config import PRESS_LEVEL, N, REGION, REGION_NAME, PATH_DATA, TIME_WINDOW, TRACKING_DISTANCE, DATA_RESOLUTION, R

LATITUDE_NORTH, LATITUDE_SOUTH, LONGITUDE_WEST, LONGITUDE_EAST = REGION[REGION_NAME]

#load data
ds_GP = xr.open_dataset(PATH_DATA + f'ds_Z{PRESS_LEVEL}_GP1-{N}.nc')
# SUBSET ... select lat-lon extent for the analysis
ds_region = ds_GP.sel(lat=slice(LATITUDE_NORTH, LATITUDE_SOUTH))

#for peak tracking, we look a little outside the region too - 25 degrees W and E past 
#the region boundaries - if that falls behind -180W or 180E, the corresponding data are copied
#and the dataset is artificially extended for the purposes of the speed calculation

ds_region_ext = extend_ds(ds_region)

def peaks_wrapper(Z):
    peaks = find_peaks(Z.copy())[0] #copy() because otherwise something is wrong with dask and memory, idk
    #the output vector should always have lenght N along the new dim called 'lon_peaks'
    while len(peaks) < 2*N: #GP1-N is a sum of N harmonic waves - there cannot be more than N peaks
        #append a negative number ... the valid values are strictly non-negative (array indices)
        #during analysis, check for -1 and drop from the list of peaks
        peaks = np.append(peaks, -1) 

    return peaks

def peaks(Z):
    ds_peaks = xr.apply_ufunc(
        peaks_wrapper, 
        Z,
        input_core_dims=[['lon']], #looking for peaks along the lon dimension
        output_core_dims=[['lon_peak']], #the output should have a new dimension
        vectorize=True     
        )
    ds_peaks = ds_peaks.rename({'Z': 'peaks'})
    return ds_peaks

#returns a ds with the indices of peaks of GP for each lat and time
#to get the real longitude values of the peaks, do something like ds_subset.lon[peak_ixs]
#smarter to do return array indices, as purely positive values are easier to work with

#dims time, lat, peak_lon (even though it's not actually exactly the longitude value)


#pass the array on the first day, the date and lat .. then generate a
#time window of length D, extract peak data during the time window at lat
#and track the peaks
def speed(peaks, day, lat, full_ds):
    day_start = day + np.timedelta64(1, 'D') #variable peaks already contains data from the first day of the tracking window
    day_end = day_start + np.timedelta64(TIME_WINDOW - 2 , 'D')  
   
    days_range = (day_start, day_end)
    #remove negative values 
    peaks = peaks[~(peaks==-1)]

    #if peaks is empty - no peak to track
    if len(peaks) == 0:
        return np.nan
    
    speed_arr = []
    #track individual peaks and save their speed
    for peak in peaks:
        s_peak = track_peak(peak, days_range, lat, full_ds)
        if not np.isnan(s_peak):
            speed_arr.append(s_peak)
    #return the average speed of the tracked peaks
    #TODO save somewhere how many peaks are being tracked on each day .. len(speed_arr)
    #return nan if speed_arr is empty - no peak tracked
    if len(speed_arr) == 0:
        return np.nan # watch out, this might cause some trouble in the future - np.nan conversion to float
    else:
        return np.mean(speed_arr)
    
def check_peak(peak_arr):
    #peak_arr either empty, len 1, len 2 or len>2
    #positive case: return int
    #negative case: return np.nan
    if len(peak_arr) == 0 or len(peak_arr) > 2:
        return np.nan
    if len(peak_arr) == 2:
        return 0 #no movement if two peaks were suddenly tracked TODO:think if this is a good idea
    if len(peak_arr) == 1:
        return peak_arr[0]

#peak must be tracked succesfully on N consecutive days in order to be used for the speed calculation
def track_peak(peak, days_range, lat, ds):
    #track a single peak, starting at day_start, ending day_end
    day_start, day_end = days_range
    #peaks on N-1 days following the initial day_start
    peaks_subset = ds.sel(lat=lat, time=slice(day_start, day_end)).peaks.to_numpy()
    tracked_peaks = [peak] #tracking starts with the original peak
    #iterate through days
    for peaks_day in peaks_subset:
        peaks_day = peaks_day[~(peaks_day==-1)]
        previous_peak = tracked_peaks[-1] #tracking the last peak
        #find peak(s) that are within the specified distance east or west from the tracked peak
        close_enough = (peaks_day >= previous_peak-TRACKING_DISTANCE) & (peaks_day <= previous_peak+TRACKING_DISTANCE)
        #apply the mask and check the result
        following_peak = check_peak(peaks_day[close_enough])

        #if previous_peak was not tracked succesfully
        #break loop and return np.nan
        if np.isnan(following_peak):
            return following_peak
        #else append the index of following_peak
        else:
            tracked_peaks = np.append(tracked_peaks, following_peak)
    
    #peak was tracked on all days of the time window
    #calculate how far it traveled on each day ... tracked_peaks[i+1] - tracked_peaks[i]
    distance_traveled = np.diff(tracked_peaks)
    #calculate the mean movement velocity of the peak during the tracking period
    #REMEMBER that the peak value is actually the index of the longitude array where the peak was located
    peak_speed = np.mean(distance_traveled) * DATA_RESOLUTION #units ... [degree longitude / day]
    #to get speed in km/day, calculate the distance traveled around a circle of latitude 
    peak_speed_km = peak_speed / 360 * 2 * np.pi * R * np.cos(np.deg2rad(lat)) #units ... [km/day]
    return peak_speed_km

def time_offset(N):
    if N % 2 == 0:
        return int(N/2)
    else:
        return int((N-1)/2)

def speed_wrapper(peaks, lat, day, full_ds):
    return speed(peaks, day, lat, full_ds)

#output ds_speed dims: time, lat
def get_speed_ds(ds):
    '''
    ds: GP1-N dataset
    returns: GP1-N speed dataset
    The function locates the peaks and troughs of the approximate geopotential field and 
    estimates its propagation speed by averaging the propagation speed of the peaks and troughs.
    '''
    #get peaks and troughs of ds
    ds_peaks = peaks(ds)
    ds_troughs = peaks(-1 * ds)
    #the last N days of the dataset can't be used for peak tracking
    #because we don't have enough days to track
    ds_peaks_N = ds_peaks.sel(time=ds.time[:-(N-1)]) 
    ds_troughs_N = ds_troughs.sel(time=ds.time[:-(N-1)]) 

    #track peaks
    ds_speed_peaks = xr.apply_ufunc(speed_wrapper,
                              ds_peaks_N,
                              ds_peaks_N['lat'],
                              ds_peaks_N['time'],
                              input_core_dims=[['lon_peak'], [], []],
                              vectorize=True,
                              kwargs={'full_ds' : ds_peaks} #the full dataset will be needed for tracking
    )
    #track troughs
    ds_speed_troughs = xr.apply_ufunc(speed_wrapper,
                              ds_troughs_N,
                              ds_troughs_N['lat'],
                              ds_troughs_N['time'],
                              input_core_dims=[['lon_peak'], [], []],
                              vectorize=True,
                              kwargs={'full_ds' : ds_troughs} #the full dataset will be needed for tracking
    )
    
    #average the trough and peek speed
    #on days when there's 0 in one dataset and a nonzero value in the other dataset
    #do not average
    zeros = (ds_speed_peaks==0) or (ds_speed_troughs==0)
    ds_speed = (ds_speed_peaks + ds_speed_troughs) / 2
    ds_speed = ds_speed.where(~zeros, ds_speed * 2)
    timedelta_speed = np.timedelta64(time_offset(N), 'D')
    ds_speed['time'] = ds_speed['time'] + timedelta_speed
    ds_speed = ds_speed.rename({'peaks':'speed'}).assign_attrs({'Description':f'GP1-{N} speed at {PRESS_LEVEL}hPa', 'Units':'[km/day]', 'Number of tracking days':TIME_WINDOW, 
                                                                'Maximal peak distance (deg longitude)':TRACKING_DISTANCE*DATA_RESOLUTION, })
    return ds_speed

ds_speed = get_speed_ds(ds_region_ext)

#linear interpolation to fill in missing values, if there are any
ds_speed = ds_speed.interpolate_na(dim='time', use_coordinate=True)


if not os.path.exists(PATH_DATA):
    os.makedirs(PATH_DATA)

ds_speed.to_netcdf(PATH_DATA + f'ds_speed_Z{PRESS_LEVEL}_GP1-{N}_{TIME_WINDOW}_days.nc')


