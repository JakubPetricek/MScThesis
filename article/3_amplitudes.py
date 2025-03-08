import xarray as xr
import numpy as np
import os
from auxiliary_scripts import extend_ds

from config import PRESS_LEVEL, N, REGION, REGION_NAME, PATH_DATA

LATITUDE_NORTH, LATITUDE_SOUTH, LONGITUDE_WEST, LONGITUDE_EAST = REGION[REGION_NAME]


ds_GP = xr.open_dataset(PATH_DATA + f'ds_Z{PRESS_LEVEL}_GP1-{N}.nc')

#########
# SUBSET ... select lat extent for the analysis
# next, extend the dataset past the region boundary in the lon direction
#########
ds_region = ds_GP.sel(lat=slice(LATITUDE_NORTH, LATITUDE_SOUTH))
ds_region_ext = extend_ds(ds_region)

#for each day, calculate ptp amplitude along a selected part of a circle of latitude 
#that is, ptp amp. across longitudes
def ptp(Z):
    return xr.apply_ufunc(
        np.ptp,
        Z,
        input_core_dims=[['lon']],
        #array was transposed so that the core dimension is at the end
        kwargs={'axis':-1} 
    )

ds_ptp = ptp(ds_region_ext)
ds_ptp = ds_ptp.rename_vars({'Z':'ptp'})
ds_ptp = ds_ptp.assign_attrs({'description':'peak to peak amplitude', 'units':'m', 'pressure level [hPa]':PRESS_LEVEL, 'number of waves':N })

#save the amplitude dataset
if not os.path.exists(PATH_DATA):
    os.makedirs(PATH_DATA)

ds_ptp.to_netcdf(PATH_DATA + f'ds_ptp_Z{PRESS_LEVEL}_GP1-{N}.nc')



