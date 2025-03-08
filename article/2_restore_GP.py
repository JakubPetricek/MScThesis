'''
Script reconstructing the geopotential height field from the calculated Fourier coefficients.
The summation of the first N terms of the Fourier series is implemented via matrix multiplication.

Input: Fourier coefficients dataset ds_Z{PRESS_LEVEL}_Fourier_coefficients1-{N}.nc
Output: Approximate geopotential height of the selected pressure level. Sum of the first N harmonic waves for all times, lats and lons.
        ds_Z{PRESS_LEVEL}_GP1-{N}.nc
'''
import numpy as np
import xarray as xr
import os

from config import L, N, PRESS_LEVEL, REGION, REGION_NAME, DATA_RESOLUTION, PATH_DATA

LATITUDE_NORTH, LATITUDE_SOUTH, LONGITUDE_WEST, LONGITUDE_EAST = REGION[REGION_NAME]

#restore GP across all longitudes 
#the spatial restriction only applies when doing the actual analysis
LON = np.arange(-180, 180+DATA_RESOLUTION, DATA_RESOLUTION)

ds_coeff = xr.open_dataset(PATH_DATA + f'ds_Z{PRESS_LEVEL}_Fourier_coefficients1-{N}.nc')

#N harmonic waves for all longitude values
def fourier_matrix(k):
    A_list = np.array([np.cos(2/L * np.pi * i * LON) for i in range(k+1)])
    A_list[0] = A_list[0] / 2 # zeroth term of the Fourier series is a_0 / 2
    B_list = np.array([np.sin(2/L * np.pi * (i+1) * LON) for i in range(k)])

    return np.concatenate([A_list, B_list])

#this function reconstructs the approximate geopotential height on one given day, for the given lat-lon range
#it is being called automatically by xr.apply_ufunc() in restore_Z()
def coeffs_to_Z(*coeffs):
    #coeffs shape (2N+1, lat)
    f_matrix = fourier_matrix(N) # shape (2N+1, lon)
    #matrix multiplication: (lat, 2N+1) @ (2N+1, lon) --> (lat, lon)
    Z_latlon = np.array([*coeffs]).T @ f_matrix 

    return Z_latlon

#get all the data variables -- all Fourier coefficients
#pass it to xr.apply_ufunc() that selects a single vector (vectorize=True) along the lat dimension (input_core_dims)
#the ouput is two dimensional (output_core_dims)... reconstructed geopotential height field on a given day
def restore_Z(dataset):
    vars = list(dataset.data_vars.values())
    Z = xr.apply_ufunc(
        coeffs_to_Z,
        *vars,
        vectorize=True,
        input_core_dims=[['lat'] for _ in vars],
        output_core_dims=[['lat', 'lon']],
        dask='allowed'
    )
    #assign coordinates to the existing dimension lon
    return Z.assign_coords({'lon' : LON})

GP_restored = restore_Z(ds_coeff)
GP_restored = GP_restored.to_dataset(name='Z')
GP_restored.assign_attrs({'description':f'geopotential height approximation', 'units':'m', 'pressure level [hPa]':PRESS_LEVEL, 'number of waves':N })

#save dataset
if not os.path.exists(PATH_DATA):
    os.makedirs(PATH_DATA)


GP_restored.to_netcdf(PATH_DATA + f'ds_Z{PRESS_LEVEL}_GP1-{N}.nc')


