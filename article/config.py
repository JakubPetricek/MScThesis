'''
This is a config file for the whole analysis. 
'''
import numpy as np
#physical constants: gravitational acc., period of Z, mean Earth radius
#do not change
L = 360 #degrees longitude
g = 9.8 #ms-2
R = 6371 #km
#input - ECMWF data in GRB format
PATH_GRB = '/cluster/projects/nn9348k/Jakub/article/data/daily_Z/'
PATH_GRB_T = '/cluster/projects/nn9348k/Jakub/article/data/6hourly_T/'
DATA_RESOLUTION = 0.5 #degrees latitude, longitude


#pressure level 
#available options: 300, 500, 850, and 1000 hPa
PRESS_LEVEL = 500

#number of terms in the approximating Fourier series
N = 7

#speed analysis
#number of days to track the peaks
TIME_WINDOW = 5
TRACKING_DISTANCE = 40 #20 degrees longitude if the resolution is 0.5

#list of different regions
#numbers in the array correspond to: 
#latitude north, latitude south, longitude west, longitude east
#in this order 
DEFAULT = {'Default' : [65, 35, -180, 180]}
EUROPE = {'Europe' : [65, 35, -15, 30]}
W_ASIA = {'W_Asia' : [65, 35, 30, 80]}
E_ASIA = {'E_Asia' : [65, 35, 90, 140]}
W_AMERICA = {'W_America' : [65, 35, -140, -100]}
E_AMERICA = {'E_America' : [65, 35, -95, -55]}

#choose spatial extent for the analysis
#change REGION to be one of the defined analysis regions
############################
REGION = E_ASIA
############################

#extract region name
REGION_NAME = list(REGION.keys())[0]


#path where to save the analysis data for given region and pressure level
ANALYSIS_FOLDER = '/cluster/projects/nn9348k/Jakub/article/'
#for data related to geopotential
PATH_DATA = ANALYSIS_FOLDER + f'data/{REGION_NAME}/{PRESS_LEVEL}hPa/{N}_Fourier/'
#for data related to temperature, depends only on region
PATH_DATA_HW = ANALYSIS_FOLDER + f'data/{REGION_NAME}/HW_data/'

#####
#Heatwave analysis
#####
START_CLIMATOLOGY = 1979
END_CLIMATOLOGY = 2009
#percentage of land grid point that have to be hotter than the threshold
SPATIAL_EXTENT = 20 

#heatwaves will be identified separately in these predefined latitude bands within the selected longitude region
#results will be also plotted for these latitude bands
LAT_BANDS = [slice(65, 55), slice(55, 45), slice(45, 35)]
#LAT_BANDS = [slice(65, 60), slice(60, 55), slice(55, 50), slice(50, 45), slice(45, 40), slice(40, 35)]

#how many Monte Carlo simulations
MC = 10000
#confidence level 99%
CONFIDENCE = 0.01

#seasons
SUMMER = {'summer':[6,7,8]}
WINTER = {'winter':[12, 1, 2]}

#choose season
SEASON = SUMMER
SEASON_NAME = list(SEASON.keys())[0]

#choose which years belong to the early/late period
early_period = np.arange(1979, 2010)
late_period = np.arange(2010, 2023)


#where to save the results
#SAVE = True to save the plots in the results folder
SAVE = True
PATH_RESULTS = PATH_DATA + f'results/{SEASON_NAME}/'


