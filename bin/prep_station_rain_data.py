import os
import xarray as xr
import glob
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import sys
import pandas as pd
import numpy as np
import datetime as dt
import cmocean
from scipy.ndimage import label
import progressbar
import argparse
#cache = '/g/data/tp28/dev/eh6215/added-value-evaluation/rundir/stations.csv'
    

def parse_arguments():
    # User argument input
    parser = argparse.ArgumentParser(description='Convert station data to CSV')
    parser.add_argument("--datapath", dest='datapath', nargs='?', type=str, \
                        default="/g/data/dp9/reana/obs_data/aus_local/station_daily/DC02D_99999999_9521826/")
    parser.add_argument("--file_format",dest='file_format',nargs='?',type=str,default= "DC02D_{0}_999999999521826.txt")
    parser.add_argument("--outpath", dest='outpath',type=str)
    parser.add_argument("--quality",  dest='quality', nargs='?', type=float, default=0.95)
    parser.add_argument("--cache", dest='cache', nargs='?', default=None)
    parser.add_argument("--lonmin", dest='lonmin',nargs='?',type=float,default=100.)
    parser.add_argument("--lonmax", dest='lonmax',nargs='?',type=float,default=155.)
    parser.add_argument("--latmin", dest='latmin',nargs='?',type=float,default=-45.)
    parser.add_argument("--latmax", dest='latmax',nargs='?',type=float,default=-10.)
    parser.add_argument("--year0", dest='year0',nargs='?',type=int,default=1990)
    parser.add_argument("--year1", dest='year1',nargs='?',type=int,default=2014)

    return parser
 
def tryconvert(value, default, *types):
    """
    Converts string to types if possible
    """
    for t in types:
        try:
            return t(value)
        except (ValueError, TypeError):
            continue
    return default

def date_parser(x):
    """
    Converts date of type YYYY/MM/DD **** to datetime object (hour and minute information is not retained)
    """
    y = x.split('/')
    return dt.datetime(int(y[2]),int(y[1]),int(y[0]))




def get_station_data(datapath,file_format,outpath,quality=0.95,cachepath=None,lonmin=100,lonmax=155,latmin=-45,latmax=-10,year0=1990,year1=2015):
    """
    Makes and caches or reads a list of stations with quality and location thresholded rainfall data
   
    Parameters
    ----------
    datapath: string
        Path to station CSV data
    file_format: string
        Format of CSV filenames
    outpath: string
        path to resultant netcdf file
    quality: float
        Threshold for fraction of data with high quality precipitation
    cachepath: string
        Path to cached station list (or to cache station list to if this doesn't exist)
    lonmin: float
        minimum station longitude to keep
    lonmax: float
        maximum station longitude to leep
    latmin: float
        minimum station latitude to keep
    latmax: float
        maximum station latitude to keep
    year0: int
        start year for station and quality filtering
    year1: int
        end year for station and quality filtering
    
    Returns
    ------
    pd.DataFrame
        Dataframe of station information
    """
    cache_read = False
    if cachepath is not None and os.path.exists(cachepath):
        # load from cache if possible
        stations = pd.read_csv(cachepath,index_col=0)
        cache_read=True
    else:
        print('No Cache')
    # load station information
        stations= pd.read_csv(os.path.join(datapath,file_format.format('StnDet')))
    # filter by requested time and space bounds
        stations = stations[stations['First year of data supplied in data file']<=year0]
        stations = stations[stations['Last year of data supplied in data file']>=year1]
        stations = stations[stations['Longitude to 4 decimal places in decimal degrees']>float(lonmin)]
        stations = stations[stations['Longitude to 4 decimal places in decimal degrees']<float(lonmax)]
        stations = stations[stations['Latitude to 4 decimal places in decimal degrees']<float(latmax)]
        stations = stations[stations['Latitude to 4 decimal places in decimal degrees']>float(latmin)]
    stations_keep = []
    obsdata = {}
    # loop over stations
    for i in progressbar.progressbar(range(len(stations))):
        station = stations['Bureau of Meteorology Station Number'].iloc[i]
        # load station data
        obs = pd.read_csv(os.path.join(datapath,file_format.format('Data_%06d'%station)),parse_dates = ["Day Month Year in DD/MM/YYYY format"],date_parser=date_parser)
        # extract relevant period
        obs = obs[obs["Day Month Year in DD/MM/YYYY format"].apply(lambda t: ((t-dt.datetime(year0,1,1)).days >= 0 and (t-dt.datetime(year1,12,31)).days <= 0))]
        # extract quality infomation 
        Q = obs['Quality of precipitation value']
        stations_keep.append((Q=='Y').mean())
        if (Q=='Y').mean() > quality:
            # extract precip if threshold is met
            P =  obs['Precipitation in the 24 hours before 9am (local time) in mm']
            P.index= obs["Day Month Year in DD/MM/YYYY format"]
            # convert to float if possible, otherwise make nan
            P=P.apply(lambda x:tryconvert(x,np.nan,float))
            # convert missing data to nan
            P[P==-9999]=np.nan
            obsdata[station] = P
            if len(obsdata)==1:
                station0 = station
            # check that all data read in has the same time index
            if len(obsdata[station]) < len(obsdata[station0]):
               obsdata[station] = obsdata[station]+0*obsdata[station0]
            assert (obsdata[station].index == obsdata[station0].index).all()

    # apply quality threshold to station list
    stations_keep=np.array(stations_keep)
    stations=stations[stations_keep>quality]
    # extract longitude and latitude
    lat = np.array(stations['Latitude to 4 decimal places in decimal degrees'])
    lon = np.array(stations['Longitude to 4 decimal places in decimal degrees'])
    if cachepath is not None and not cache_read:
        # save to cache if requested and not already sourced from cache
        stations.to_csv(cachepath)    
    # convert precip data to xarray
    obs = xr.DataArray(np.array([obsdata[key][obsdata[key].index.to_series()].values for key in obsdata]).T,coords = {'time':obsdata[station0].index.to_series(),'station':list(obsdata.keys())},name='pr')
    # assign spatial coordinates and units
    obs = obs.assign_coords(lon = ('station',lon),lat = ('station',lat))
    obs.attrs['units'] = 'mm'
    # save as netcdf file (TODO: compression)
    obs.to_netcdf(outpath)
    return obs

if __name__=='__main__':
    parser = parse_arguments()
    args   = parser.parse_args()
    data   = get_station_data(args.datapath, args.file_format, args.outpath, args.quality, args.cache, args.lonmin, args.lonmax, args.latmin, args.latmax, args.year0, args.year1)
