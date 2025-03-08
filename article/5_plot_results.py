import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
import tikzplotlib

from config import PATH_DATA, PATH_RESULTS, PRESS_LEVEL, N, REGION_NAME, LAT_BANDS, TIME_WINDOW, MC, CONFIDENCE, SAVE
from config import SEASON, SEASON_NAME

#compare early-period speeds vs. late-period speeds
from config import early_period, late_period
early_start = early_period[0]
early_end = early_period[-1]
late_start = late_period[0]
late_end = late_period[-1]

#this thing is here because of saving plots using tikzplotlib
#as a .tex file that can be included into a latex document
#it's some error workaround I found on github
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
Line2D._us_dashSeq    = property(lambda self: self._dash_pattern[1])
Line2D._us_dashOffset = property(lambda self: self._dash_pattern[0])
Legend._ncol = property(lambda self: self._ncols)
########

#load data 
#ptp
ds_ptp = xr.open_dataset(PATH_DATA + f'ds_ptp_Z{PRESS_LEVEL}_GP1-{N}.nc')
ds_ptp_HW = xr.open_dataset(PATH_DATA + f'ds_HW_ptp_Z{PRESS_LEVEL}_GP1-{N}.nc')
ds_ptp_outside_HW = xr.open_dataset(PATH_DATA + f'ds_outside_HW_ptp_Z{PRESS_LEVEL}_GP1-{N}.nc')
#speed
ds_speed = xr.open_dataset(PATH_DATA + f'ds_speed_Z{PRESS_LEVEL}_GP1-{N}_{TIME_WINDOW}_days.nc')
ds_speed_HW = xr.open_dataset(PATH_DATA + f'ds_HW_speed_Z{PRESS_LEVEL}_GP1-{N}_{TIME_WINDOW}_days.nc')
ds_speed_outside_HW = xr.open_dataset(PATH_DATA + f'ds_outside_HW_speed_Z{PRESS_LEVEL}_GP1-{N}_{TIME_WINDOW}_days.nc')

#season
SEASON_MONTHS = SEASON[SEASON_NAME]

ds_ptp = ds_ptp.sel(time=ds_ptp.time.dt.month.isin(SEASON_MONTHS))
ds_ptp_HW = ds_ptp_HW.sel(time=ds_ptp_HW.time.dt.month.isin(SEASON_MONTHS))
ds_ptp_outside_HW = ds_ptp_outside_HW.sel(time=ds_ptp_outside_HW.time.dt.month.isin(SEASON_MONTHS))

ds_speed = ds_speed.sel(time=ds_speed.time.dt.month.isin(SEASON_MONTHS))
ds_speed_HW = ds_speed_HW.sel(time=ds_speed_HW.time.dt.month.isin(SEASON_MONTHS))
ds_speed_outside_HW = ds_speed_outside_HW.sel(time=ds_speed_outside_HW.time.dt.month.isin(SEASON_MONTHS))


####
#Pass the same lat band of 3 datasets to this function: 
#the two studied (subset) datasets (e. g. ds_hw and ds_non_hw)
#and a third dataset that is a union of the two previous ones
#Returns 3 arrays: percentiles 1 through 99 of ds1, ds2, and the MC simulation results
####
def get_quantiles(ds1, ds2, ds_original):
    #get the DataArray name, and the number of non-NaN samples in each ds
    DA_name = list(ds1.data_vars)[0]
    ds1_nr_samples = np.sum(~np.isnan(ds1[DA_name])).to_numpy()
    ds2_nr_samples = np.sum(~np.isnan(ds2[DA_name])).to_numpy()

    #calculate only percentiles 1 through 99
    #perc. 0 and 100 are the min/max values of the respective ds
    ds1_perc = np.nanpercentile(ds1[DA_name], np.linspace(1, 99, 99))
    ds2_perc = np.nanpercentile(ds2[DA_name], np.linspace(1, 99, 99))

    #absolute distance from the line y=x
    #null hypothesis: no change between ds1 and ds2 -> real_distance is zero for all perc.
    real_distance = np.abs(ds1_perc - ds2_perc)

    #array of the same lenght as ds_perc
    #percentage of how many times was the simulated distance greater than the real distance
    percentile_significance = monte_carlo(ds_original, real_distance, ds1_nr_samples, ds2_nr_samples)
    
    return ds1_perc, ds2_perc, percentile_significance


def monte_carlo(ds_original, real_distance, ds1_nr_samples, ds2_nr_samples, MC=MC):
    DA_name = list(ds_original.data_vars)[0]
    flat_ds = ds_original[DA_name].to_numpy().flatten()
    significance = np.zeros_like(real_distance)

    for _ in range(MC):
        #from the flat dataarray, draw the same number of samples as when calculating dsX_perc
        random_subset_1 = np.random.choice(flat_ds, size=ds1_nr_samples)
        random_subset_2 = np.random.choice(flat_ds, size=ds2_nr_samples)

        ds1_perc = np.nanpercentile(random_subset_1, np.linspace(1, 99, 99))
        ds2_perc = np.nanpercentile(random_subset_2, np.linspace(1, 99, 99))

        simulated_distance = np.abs(ds1_perc - ds2_perc)

        #check if the simulated distance of each percentile from the axis y=x is
        #greater between the random percentiles
        significance[simulated_distance > real_distance] += 1
    
    #express significance as a percentage of all the simulated cases
    significance = significance / MC

    return significance


def plot_QQ(ds1_perc, ds2_perc, significance, plot_description, save_plot=False):

    max = np.nanmax([ds1_perc[-1], ds2_perc[-1]])  #larger of the two 99th percentiles
    min = np.nanmin([ds1_perc[0], ds2_perc[0]]) #smaller of the two 1st percentiles
    #plot the quadrant axis
    #were the two underlying data distributions the same
    #the scatter points would end up directly on the line  
    plt.plot([min, max], [min, max], linestyle='--', color='gray') 

    #significance denoted by color
    #statistically sig - red, otherwise blue
    #99% confidence level
    color = np.where(significance<=CONFIDENCE, 'r', 'b')
    plt.scatter(ds1_perc, ds2_perc, c=color, alpha=0.5, marker='.', s=75)
    
    #to make the plot readable, plot several deciles differently
    deciles = np.arange(50,100,10)
    for dec in deciles:
        plt.scatter(ds1_perc[dec-1], ds2_perc[dec-1], c=color[dec-1], marker='D', s=55) 

    title, DA_name, plot, xlabel, ylabel, filename = plot_description
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.grid()

    if save_plot:
        if not os.path.exists(PATH_RESULTS + f'{DA_name}/{plot}/'):
            os.makedirs(PATH_RESULTS + f'{DA_name}/{plot}/')

        #save as svg for simple preview
        plt.savefig(PATH_RESULTS + f'{DA_name}/{plot}/' + filename + '.svg')  

        #save as a .tex file to include in a Latex document
        tikzplotlib.save(PATH_RESULTS + f'{DA_name}/{plot}/' + filename + '.tex')
        plt.close()
    else:
        plt.show()


#####
#plot PtP results
#####
#pass the full datasets to this function
#and loop over the latitude slices given in lat array
#plot_type is a dictionary describing what kind of results we're plotting
#such as: 'hw-nonhw', 'early-late', 'earlyhw-latehw'
#this dictionary also includes xlabel, ylabel
#save=True if you want to save the pictures to PATH_RESULTS
def plot_func(ds1, ds2, ds_original, lat, plot_type, save=False):
    DA_name = list(ds_original.data_vars)[0]

    for l in lat:
        ds1_lat = ds1.sel(lat=l)
        ds2_lat = ds2.sel(lat=l)
        ds_orig_lat = ds_original.sel(lat=l)

        #plot description
        plot = plot_type['plot']
        xlabel = plot_type['xlabel']
        ylabel = plot_type['ylabel']
        lat_slice = f'{int(ds1_lat.lat.values[-1])}°-{int(ds1_lat.lat.values[0])}°N'
        filename = f'QQ_{DA_name}_{plot}_{lat_slice}'
        title = f'{DA_name} {plot} at {lat_slice}, region {REGION_NAME}'
        plot_description = [title, DA_name, plot, xlabel, ylabel, filename]

        ds1_perc, ds2_perc, percentile_significance = get_quantiles(ds1_lat, ds2_lat, ds_orig_lat)
        #plot the results or save to PATH_RESULTS 
        plot_QQ(ds1_perc, ds2_perc, percentile_significance, plot_description, save)


#####
#plot PtP results
#####
#compare heatwave amplitudes vs. non-heatwave amplitudes
plot_1_ptp = {'plot':'hw-nonhw', 'xlabel':'quantiles of non-HW PtP amp. [m]', 'ylabel':'quantiles of HW PtP amp. [m]'}
#y>x ... increased amplitude during HWs
plot_func(ds_ptp_outside_HW, ds_ptp_HW, ds_ptp, LAT_BANDS, plot_1_ptp, save=SAVE)


ds_ptp_early = ds_ptp.sel(time=ds_ptp.time.dt.year.isin(early_period))
ds_ptp_late = ds_ptp.sel(time=ds_ptp.time.dt.year.isin(late_period))
plot_2_ptp = {'plot':'early-late', 'xlabel':f'quantiles of {early_start}-{early_end} PtP amp. [m]', 'ylabel':f'quantiles of {late_start}-{late_end} PtP amp. [m]'}
#y>x ... amplitudes increased over time
plot_func(ds_ptp_early, ds_ptp_late, ds_ptp, LAT_BANDS, plot_2_ptp, save=SAVE)

#compare early-period heatwave amplitudes vs. late-period heatwave amplitudes
ds_ptp_HW_early = ds_ptp_HW.sel(time=ds_ptp_HW.time.dt.year.isin(early_period))
ds_ptp_HW_late = ds_ptp_HW.sel(time=ds_ptp_HW.time.dt.year.isin(late_period))
plot_3_ptp = {'plot':'earlyHW-lateHW', 'xlabel':f'{early_start}-{early_end} PtP amp. during HWs [m]', 'ylabel':f'{late_start}-{late_end} PtP amp. during HWs [m]'}
plot_func(ds_ptp_HW_early, ds_ptp_HW_late, ds_ptp_HW, LAT_BANDS, plot_3_ptp, save=SAVE)


####
#plot speed results
####
#compare heatwave speed vs. non-heatwave speed
plot_1_speed = {'plot':'hw-nonhw', 'xlabel':'quantiles of non-HW speed [km/day]', 'ylabel':'quantiles of HW speed [km/day]'}
plot_func(ds_speed_outside_HW, ds_speed_HW, ds_speed, LAT_BANDS, plot_1_speed, save=SAVE)



ds_speed_early = ds_speed.sel(time=ds_speed.time.dt.year.isin(early_period))
ds_speed_late = ds_speed.sel(time=ds_speed.time.dt.year.isin(late_period))
plot_2_speed = {'plot':'early-late', 'xlabel':f'quantiles of {early_start}-{early_end} speed [km/day]', 'ylabel':f'quantiles of {late_start}-{late_end} speed [km/day]'}
plot_func(ds_speed_early, ds_speed_late, ds_speed, LAT_BANDS, plot_2_speed, save=SAVE)

#compare early-period heatwave speeds vs. late-period heatwave speeds
ds_speed_HW_early = ds_speed_HW.sel(time=ds_speed_HW.time.dt.year.isin(early_period))
ds_speed_HW_late = ds_speed_HW.sel(time=ds_speed_HW.time.dt.year.isin(late_period))
plot_3_speed = {'plot':'earlyHW-lateHW', 'xlabel':f'{early_start}-{early_end} speed during HWs [km/day]', 'ylabel':f'{late_start}-{late_end} speed during HWs [km/day]'}
plot_func(ds_speed_HW_early, ds_speed_HW_late, ds_speed_HW, LAT_BANDS, plot_3_speed, save=SAVE)


"""
ds_ptp = xr.open_dataset(PATH_DATA + f'ds_ptp_Z{PRESS_LEVEL}_GP1-{N}.nc')
ds_ptp_HW = xr.open_dataset(PATH_DATA + f'ds_HW_ptp_Z{PRESS_LEVEL}_GP1-{N}.nc')
ds_ptp_outside_HW = xr.open_dataset(PATH_DATA + f'ds_outside_HW_ptp_Z{PRESS_LEVEL}_GP1-{N}.nc')
#speed
ds_speed = xr.open_dataset(PATH_DATA + f'ds_speed_Z{PRESS_LEVEL}_GP1-{N}_{TIME_WINDOW}_days.nc')
ds_speed_HW = xr.open_dataset(PATH_DATA + f'ds_HW_speed_Z{PRESS_LEVEL}_GP1-{N}_{TIME_WINDOW}_days.nc')
ds_speed_outside_HW = xr.open_dataset(PATH_DATA + f'ds_outside_HW_speed_Z{PRESS_LEVEL}_GP1-{N}_{TIME_WINDOW}_days.nc')

#season
SEASON_MONTHS = [6,7,8,9]
ds_ptp = ds_ptp.sel(time=ds_ptp.time.dt.month.isin(SEASON_MONTHS))
ds_ptp_HW = ds_ptp_HW.sel(time=ds_ptp_HW.time.dt.month.isin(SEASON_MONTHS))
ds_ptp_outside_HW = ds_ptp_outside_HW.sel(time=ds_ptp_outside_HW.time.dt.month.isin(SEASON_MONTHS))

ds_speed = ds_speed.sel(time=ds_speed.time.dt.month.isin(SEASON_MONTHS))
ds_speed_HW = ds_speed_HW.sel(time=ds_speed_HW.time.dt.month.isin(SEASON_MONTHS))
ds_speed_outside_HW = ds_speed_outside_HW.sel(time=ds_speed_outside_HW.time.dt.month.isin(SEASON_MONTHS))

early_period = np.arange(1995, 2010)
late_period = np.arange(2010, 2023)

#compare heatwave amplitudes vs. non-heatwave amplitudes
plot_1_ptp = {'plot':'hw-nonhw', 'xlabel':'quantiles of non-HW PtP amp. [m]', 'ylabel':'quantiles of HW PtP amp. [m]'}
#y>x ... increased amplitude during HWs
plot_func(ds_ptp_outside_HW, ds_ptp_HW, ds_ptp, LAT_BANDS, plot_1_ptp, save=False)


ds_ptp_early = ds_ptp.sel(time=ds_ptp.time.dt.year.isin(early_period))
ds_ptp_late = ds_ptp.sel(time=ds_ptp.time.dt.year.isin(late_period))
plot_2_ptp = {'plot':'early-late', 'xlabel':f'quantiles of {early_start}-{early_end} PtP amp. [m]', 'ylabel':f'quantiles of {late_start}-{late_end} PtP amp. [m]'}
#y>x ... amplitudes increased over time
plot_func(ds_ptp_early, ds_ptp_late, ds_ptp, LAT_BANDS, plot_2_ptp, save=False)

#compare early-period heatwave amplitudes vs. late-period heatwave amplitudes
ds_ptp_HW_early = ds_ptp_HW.sel(time=ds_ptp_HW.time.dt.year.isin(early_period))
ds_ptp_HW_late = ds_ptp_HW.sel(time=ds_ptp_HW.time.dt.year.isin(late_period))
plot_3_ptp = {'plot':'earlyHW-lateHW', 'xlabel':f'{early_start}-{early_end} PtP amp. during HWs [m]', 'ylabel':f'{late_start}-{late_end} PtP amp. during HWs [m]'}
plot_func(ds_ptp_HW_early, ds_ptp_HW_late, ds_ptp_HW, LAT_BANDS, plot_3_ptp, save=False)

"""