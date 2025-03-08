import xarray as xr
from config import REGION, REGION_NAME, TRACKING_DISTANCE, DATA_RESOLUTION

LATITUDE_NORTH, LATITUDE_SOUTH, LONGITUDE_WEST, LONGITUDE_EAST = REGION[REGION_NAME]



#########
# To capture more trackable peaks, we extend the analysis in the longitudinal direction 
# so that the peaks that are withing TRACKING_DISTANCE from the borders of the region are tracked.
# This makes sense, as the peaks are either leaving or entering the region, and therefore 
# carry some information about the propagation speed in the region. 
# The same argument can be made about the PtP amplitude - the flow right outside the region
# affects the surface conditions in the region. 
#########
def extend_ds(ds):
    w = LONGITUDE_WEST - TRACKING_DISTANCE*DATA_RESOLUTION
    e = LONGITUDE_EAST + TRACKING_DISTANCE*DATA_RESOLUTION
    ds_ext = ds
    if w >= -180 and e <= 180:
        return ds.sel(lon=slice(w, e))
    
    if w <= -180:
        #copy left side of the dataset
        we = w + 180 #number of degrees from the right side of the array we want to copy
        ds_left = ds.sel(lon=slice(180 + we, 180))  
        ds_left['lon'] = ds_left['lon'] - 360 #for concat purposes
        ds_ext = xr.concat([ds_left, ds_ext], 'lon')
    else:
        ds_ext = ds_ext.sel(lon=slice(w, LONGITUDE_EAST))
    if e >= 180:
        ew = e - 180
        ds_right = ds.sel(lon=slice(-180, -180+ew))
        ds_right['lon'] = ds_right['lon'] + 360 #for concat purposes

        ds_ext = xr.concat([ds_ext, ds_right], 'lon')
    else:
        #in case the ds was already extended, update w
        w = ds_ext['lon'][0]
        ds_ext = ds_ext.sel(lon=slice(w, e))

    return ds_ext