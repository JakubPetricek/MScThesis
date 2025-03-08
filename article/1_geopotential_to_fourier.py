'''
When approximating a function by its Fourier series, it needs to be periodic,
therefore I have expanded the data so that the first and last values are the same,
and the array with longitude values also goes from -180 to 180, not to 179.5 as previously
Then, the first N Fourier coefficients are calculated and saved as variables a_0, ..., a_N, b1, ..., b_N to a dataset. 
The spatial extent (lat, lon) is reduced according to the global parameters.

Input: grb files with daily geopotential data
Output: a) ds_Z{press_lvl}_lon_extended.nc -- geopotential height of {press_lvl}hPa within selected latitudes and with extended longitude
        b) Z{press_lvl}_Fourier_coefficients1-N.nc -- the first N Fourier coeff. 
'''
import numpy as np
import xarray as xr
import os

from config import PATH_GRB, PATH_DATA, PRESS_LEVEL, N, g, L, REGION, REGION_NAME

LATITUDE_NORTH, LATITUDE_SOUTH, LONGITUDE_WEST, LONGITUDE_EAST = REGION[REGION_NAME]

#open multi-file dataset in grib format ... depends on the cfgrib package
ds = xr.open_mfdataset(PATH_GRB + '*.grb', parallel=True, combine='nested', concat_dim='time', engine='cfgrib')
ds = ds.rename({'latitude':'lat', 'longitude':'lon', 'isobaricInhPa':'press_levels', 'z':'Z'})
ds = ds.sel(press_levels = PRESS_LEVEL) #select 500hPa pressure level
ds = ds/g #fix the units --> [m]

# artificially force periodicity
copied_row = ds.sel(lon='-180')
copied_row = copied_row.assign_coords({'lon':180})
copied_row = copied_row.expand_dims(dim={'lon':1})
ds_extended = xr.combine_by_coords([ds, copied_row])
ds_extended = ds_extended.Z.assign_attrs({'description': 'geopotential height', 'units':'m', 'pressure_level (hPa)':PRESS_LEVEL}) 
ds_extended = ds_extended.drop_vars(['press_levels', 'step', 'valid_time'])

#spatial extent of the analysis
#here we can only specify the latitude slice, since we need the full longitude range for the Fourier decomposition
ds_extended = ds_extended.sel(lat=slice(LATITUDE_NORTH,LATITUDE_SOUTH))

# save longitudinaly extended datase, ready to be Fourier decomposed
ds_extended = ds_extended.to_dataset(name='Z')

#check if there is already a folder for the analysis data
if not os.path.exists(PATH_DATA):
    os.makedirs(PATH_DATA)

ds_extended.to_netcdf(PATH_DATA + f'ds_Z{PRESS_LEVEL}_lon_extended.nc')

############
# Fourier decomposition
############
lon = ds_extended.lon.to_numpy() # [-180, ..., 180]

def decompose(dataset):
    #period of Z ... one revolution around the Earth = 360 degrees
    a0 = dataset.integrate('lon') * 2/L
    a0 = a0.rename({'Z':'a0'})
    list_a0 = [a0]

    # how many coefficients a_n, b_n to calculate is determined by the imported global variable N
    coeff_a = []
    coeff_b = []

    for n in range(N):
        a_name = f'a{n+1}'
        a = (dataset * np.cos(2*np.pi/L * (n+1) * lon) * 2/L).integrate('lon') ### n+1
        a = a.rename({'Z':a_name})
        coeff_a.append(a)

        b_name = f'b{n+1}'
        b = (dataset * np.sin(2*np.pi/L * (n+1) * lon)*2/L).integrate('lon')
        b = b.rename({'Z':b_name})
        coeff_b.append(b)
        
    #concatenate coefficitens
    coeff = list_a0 + coeff_a + coeff_b

    merged = xr.merge(coeff)
    

    return merged

ds_decomposed = decompose(ds_extended)
ds_decomposed = ds_decomposed.assign_attrs({'Description':f'Fourier decomposition coefficients', 'pressure level [hPa]':PRESS_LEVEL, 'number of waves': N})


if not os.path.exists(PATH_DATA):
    os.makedirs(PATH_DATA)

ds_decomposed.to_netcdf(PATH_DATA + f'ds_Z{PRESS_LEVEL}_Fourier_coefficients1-{N}.nc')
