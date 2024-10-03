import xarray as xr
import numpy as np
import os
import dask
import datetime
import pandas as pd
import xesmf as xe
import matplotlib.pyplot as plt
import warnings
import ast
import logging
import sys
import cmdline_provenance as cmdprov
import tempfile
import lib_standards
import lib_spatial
import glob
import re
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import copy


def get_logger(name, level='info'):
    """Get a logging object.

    Args:
        name (str): Name of the module currently logging.
        level (str, optional): Level of logging to emit. Defaults to 'debug'.

    Returns:
        logging.Logger: Logging object.
    """

    logger = logging.Logger(name)
    handler = logging.StreamHandler(sys.stdout)
    level = getattr(logging, level.upper())
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

#< Get logger
loglevel = os.getenv("LIB_LOGLEVEL", default="INFO")
logger = get_logger(__name__, loglevel)


def literal_eval(v):
    if v:
        return ast.literal_eval(v)


def str2bool(v):
    """Convert str to bool

    Args:
        v (str): String to be converted

    Returns:
        bool : True/False
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def guess_bounds(dim):
    """
    Guess the dimension bounds.

    :param dim:    Input values.
    :return:       Bounds.
    """
    delta  = dim[1]-dim[0]
    bounds = np.append(dim - delta/2,dim[-1]+delta/2)
    return bounds


def rename_helper(ds, **kwargs):
    """
    Helper to rename variables or coordinates in dataset.

    :param ds: Input dataset
    :return:   Renamed dataset
    """
    for key in kwargs:
        if key in ds.coords:
            ds = ds.rename({key:kwargs[key]})
    return ds


def sel_helper(ds, **kwargs):
    """
    Helper to select coordinates in dataset.

    :param ds: Input dataset
    :return:   Renamed dataset
    """
    for key in kwargs:
        if key in ds.coords:
            ds = ds.sel({key:kwargs[key]})
    return ds


def strictly_increasing(L):
    """
    Check if input is strictly increasing.

    :param L: List of values
    :return:  True/False if strictly increasing
    """
    return all(x<y for x, y in zip(L, L[1:]))


def sortCoordsAscending(ds):
    """
    Sort the coordinates of a dataset in ascending order.

    :param ds: xarray dataset
    :return:   Sorted dataset
    """
    coords2ignore = ['forecast_reference_time', 'forecast_period', 'sigma'] # Don't worry about sorting those coordinates
    for c in ds.coords:
        if c in coords2ignore:
            continue
        if c == 'x' or c=='y': # Don't sort rotated lat and lon
            continue
        cval = ds[c].values
        if ds[c].shape: # check if it has a length (e.g. ignore single length dimensions)
            if len(ds[c].shape) == 1: #< Only for rectilinear grid (i.e. lat and lon are 1d)
                if not strictly_increasing(cval):
                    # with dask.config.set({"array.chunk-size": "128 MiB"}): #< Fixes an issue with dask where sorting re-chunks array to tiny sizes
                    ds = ds.sortby(ds[c])
    return ds


def tidy_coords(ds):
    """
    Tidy the coordinates of a dataset.

    :param ds: xarray dataset
    :param ifiles: input files
    :param rm_first: remove the first double entry
    :param rm_second: remove the second double entry
    :return:   Tidied up dataset
    """
    #< Make sure time index is unique
    if "time" in ds.coords:
        _,index = np.unique(ds['time'], return_index=True)
        if not len(index) == len(ds['time']):
            logger.warning('In tidy_coords: Duplicate time indicies found and removed!')
            ds = ds.isel(time=index)
    #< Make sure lon range is 0 to 360
    if 'longitude' in ds.coords:
        ds = ds.assign_coords(longitude=(ds['longitude'] % 360))
    #< Make sure coordinates are ascending order
    ds = sortCoordsAscending(ds)
    return ds


def convert_units(ds):
    """
    Convert to standard units (e.g., Kelvin)

    :param ds: xarray dataset
    :return: Dataset
    """

    logger.info("Checking units")
    for key in ds:
        if "units" in ds[key].attrs:
            if ds[key].attrs["units"] == "degrees_Celsius":
                ds[key] = ds[key] + 273.15
                ds[key].attrs["units"] = "K"
                logger.debug("Converting units from degrees_Celsius to K")

            elif ds[key].attrs["units"] in ["kg m-2 s-1", "kg m**-2 s**-1"]:
                ds[key] = ds[key] * 86400
                ds[key].attrs["units"] = "mm"
                logger.debug("Converting units from kg m-2 s-1 to mm")
    return ds



def open_dataset(ifiles, **kwargs):
    """Open ifiles with xarray and return dataset

    Args:
        ifiles (str): Path/File to be opened
        kwargs (dict): Keyword arguments to be passed to xarray.open_mfdataset

    Returns:
        xarray.Dataset
    """
    read_kwargs = {'combine':'nested', 'concat_dim':'time', 'parallel':True, 'coords':'minimal', 'data_vars':'minimal', 'compat':'override', 'preprocess':None}
    for key in read_kwargs:
        if key in kwargs:
            read_kwargs[key] = kwargs[key]
    if not ifiles: # check if list is empty
        logger.error(f'No files for {ifiles}')
    ds = xr.open_mfdataset(ifiles, **read_kwargs)
    #< Tidy the coordinates
    ds = rename_helper(ds, **{"latitude":"lat", "longitude": "lon", "lev":"pressure"})
    ds = tidy_coords(ds)
    #< Unit conversion
    ds = convert_units(ds)

    return ds


def AVse(X_obs, X_gdd, X_rcm):
    """
    Calculate added value (AV) using the square error between the global
    driving model (gdd), the regional climate model (rcm) and observations (obs).

    :param X_obs: xarray containing the observations
    :param X_gdd: xarray containing the global driving data
    :param X_rcm: xarray containing the regional climate model
    :return:      xarray containting the AV (RMSE) for each grid-point
    """

    with xr.set_options(keep_attrs=True):
        out = ((X_gdd-X_obs)**2) - ((X_rcm-X_obs)**2)
    return out


def AVse_norm(X_obs, X_gdd, X_rcm):
    """
    Calculate combined error of the GDD and RCM using the square error between the global
    driving model (gdd), the regional climate model (rcm) and observations (obs).

    Used to normalise the AV (MSE) to be between -1 and 1.

    :param X_obs: xarray containing the observations
    :param X_gdd: xarray containing the global driving data
    :param X_rcm: xarray containing the regional climate model
    :return:      xarray containting the AV (RMSE) for each grid-point
    """

    with xr.set_options(keep_attrs=True):
        out = ((X_gdd-X_obs)**2) + ((X_rcm-X_obs)**2)
    return out


def AVmse(X_obs, X_gdd, X_rcm):
    """
    Calculate mean added value (AV) over lat/lon using the mean square error between the global
    driving model (gdd), the regional climate model (rcm) and observations (obs).

    :param X_obs: xarray containing the observations
    :param X_gdd: xarray containing the global driving data
    :param X_rcm: xarray containing the regional climate model
    :return:      xarray containting the AV (RMSE) for each grid-point
    """

    with xr.set_options(keep_attrs=True):
        out = AVse(X_obs, X_gdd, X_rcm).mean(dim=['lat','lon'])
    return out


def AVmse_norm(X_obs, X_gdd, X_rcm):
    """
    Calculate mean combined error over lat/lon of the GDD and RCM using the mean square error between the global
    driving model (gdd), the regional climate model (rcm) and observations (obs).

    Used to normalise the AV (MSE) to be between -1 and 1.

    :param X_obs: xarray containing the observations
    :param X_gdd: xarray containing the global driving data
    :param X_rcm: xarray containing the regional climate model
    :return:      xarray containting the AV (RMSE) for each grid-point
    """

    with xr.set_options(keep_attrs=True):
        out = AVse_norm(X_obs, X_gdd, X_rcm).mean(dim=['lat','lon'])
    return out


def AVrmse(X_obs, X_gdd, X_rcm):
    """
    Calculate added value (AV) using the root mean square error between the global
    driving model (gdd), the regional climate model (rcm) and observations (obs).

    :param X_obs: xarray containing the observations
    :param X_gdd: xarray containing the global driving data
    :param X_rcm: xarray containing the regional climate model
    :return:      xarray containting the AV (RMSE) for each grid-point
    """
    with xr.set_options(keep_attrs=True):
        out = np.sqrt((X_gdd-X_obs)**2) - np.sqrt((X_rcm-X_obs)**2)
    return out


def AVperkins(X_obs, X_gdd, X_rcm,spacing=1):
    """
    Calculate added value (AV) using a Perkins Skill Score between the global
    driving model (gdd), the regional climate model (rcm) and observations (obs).
    Designed for spatial distributions of percentiles but may be generalisable.

    :param X_obs: xarray containing the observations
    :param X_gdd: xarray containing the global driving data
    :param X_rcm: xarray containing the regional climate model
    :return:      xarray containting the AV (RMSE) for each grid-point
    """
    from xhistogram.xarray import histogram
    # calculate bounds
    mx = max(X_obs.max(),X_gdd.max(),X_rcm.max()).values
    mn = min(X_obs.min(),X_gdd.min(),X_rcm.min()).values
    mn = np.floor(mn/spacing)*spacing
    mx = np.ceil(mx/spacing)*spacing
    bins = np.arange(mn,mx+spacing,spacing)
    # calculate normed histograms
    hist_obs = histogram(X_obs,bins=[bins],density=1)
    hist_gdd = histogram(X_gdd,bins=[bins],density=1)
    hist_rcm = histogram(X_rcm,bins=[bins],density=1)
    # calculate perkins scores
    pss_gdd = np.minimum(hist_gdd,hist_obs).sum()
    pss_rcm = np.minimum(hist_rcm,hist_obs).sum()
    return pss_rcm - pss_gdd


def AVcorr(X_obs, X_gdd, X_rcm):
    """
    Calculate mean added value (AV) over lat/lon using the correlation error between the global
    driving model (gdd), the regional climate model (rcm) and observations (obs).

    :param X_obs: xarray containing the observations
    :param X_gdd: xarray containing the global driving data
    :param X_rcm: xarray containing the regional climate model
    :return:      xarray containting the AV (corr) for each grid-point
    """
    with xr.set_options(keep_attrs=True):
        out = (1-xr.corr(X_gdd, X_obs, dim=['lat','lon'])) - (1-xr.corr(X_rcm, X_obs, dim=['lat','lon']))
    return out


def AVcorr_norm(X_obs, X_gdd, X_rcm):
    """
    Calculate mean added value (AV) over lat/lon using the correlation error between the global
    driving model (gdd), the regional climate model (rcm) and observations (obs).

    :param X_obs: xarray containing the observations
    :param X_gdd: xarray containing the global driving data
    :param X_rcm: xarray containing the regional climate model
    :return:      xarray containting the AV (corr) for each grid-point
    """
    with xr.set_options(keep_attrs=True):
        out = (1-xr.corr(X_gdd, X_obs, dim=['lat','lon'])) + (1-xr.corr(X_rcm, X_obs, dim=['lat','lon']))
    return out


def AVbias(X_obs, X_gdd, X_rcm):
    """
    Calculate mean added value (AV) over lat/lon using the bias between the global
    driving model (gdd), the regional climate model (rcm) and observations (obs).

    :param X_obs: xarray containing the observations
    :param X_gdd: xarray containing the global driving data
    :param X_rcm: xarray containing the regional climate model
    :return:      xarray containting the AV (bias) for each grid-point
    """
    with xr.set_options(keep_attrs=True):
        out = ((np.abs(X_gdd-X_obs)) - (np.abs(X_rcm-X_obs)))
    return out


def AVbias_norm(X_obs, X_gdd, X_rcm):
    """
    Calculate mean added value (AV) over lat/lon using the bias between the global
    driving model (gdd), the regional climate model (rcm) and observations (obs).

    :param X_obs: xarray containing the observations
    :param X_gdd: xarray containing the global driving data
    :param X_rcm: xarray containing the regional climate model
    :return:      xarray containting the AV (bias) for each grid-point
    """
    with xr.set_options(keep_attrs=True):
        out = ((np.abs(X_gdd-X_obs)) + (np.abs(X_rcm-X_obs)))
    return out


def AVstd(X_obs, X_gdd, X_rcm):
    """
    Calculate mean added value (AV) over lat/lon using the standard deviation between the global
    driving model (gdd), the regional climate model (rcm) and observations (obs).

    :param X_obs: xarray containing the observations
    :param X_gdd: xarray containing the global driving data
    :param X_rcm: xarray containing the regional climate model
    :return:      xarray containting the AV (standard deviation)
    """
    with xr.set_options(keep_attrs=True):
        var_obs = X_obs.std(dim=['lat','lon']); var_gdd = X_gdd.std(dim=['lat','lon']); var_rcm = X_rcm.std(dim=['lat','lon'])
        out = np.abs(var_gdd - var_obs)  -  np.abs(var_rcm - var_obs)
    return out


def AVstd_norm(X_obs, X_gdd, X_rcm):
    """
    Calculate mean added value (AV) over lat/lon using the standard deviation between the global
    driving model (gdd), the regional climate model (rcm) and observations (obs).

    :param X_obs: xarray containing the observations
    :param X_gdd: xarray containing the global driving data
    :param X_rcm: xarray containing the regional climate model
    :return:      xarray containting the AV (standard deviation)
    """

    with xr.set_options(keep_attrs=True):
        var_obs = X_obs.std(dim=['lat','lon']); var_gdd = X_gdd.std(dim=['lat','lon']); var_rcm = X_rcm.std(dim=['lat','lon'])
        out = np.abs(var_gdd - var_obs)  +  np.abs(var_rcm - var_obs)
    return out


def AVse_frac(X_obs, X_gdd, X_rcm, thresh=0):
    """
    Calculate fraction of mean added value (AV) over lat/lon using the root mean square error between the global
    driving model (gdd), the regional climate model (rcm) and observations (obs).

    :param X_obs: xarray containing the observations
    :param X_gdd: xarray containing the global driving data
    :param X_rcm: xarray containing the regional climate model
    :return:      xarray containting the AV (RMSE) for each grid-point
    """

    with xr.set_options(keep_attrs=True):
        av = AVse(X_obs, X_gdd, X_rcm)
        #< Count how often av is larger than threshold and divide by size of non nans
        av_frac = (xr.where(av>thresh, 1, 0)).sum(['lat','lon']) / (xr.where(xr.ufuncs.isnan(av), 0, 1)).sum(['lat','lon'])
        av_frac = av_frac * 100
    return av_frac


def frac_ss(da, thresh=0):
    """
    Calculate fraction of mean added value (AV) over lat/lon using added value as input.

    :param av: Added Value
    :return:   xarray containting the AV fraction
    """

    with xr.set_options(keep_attrs=True):
        #< Count how often av is larger than threshold and divide by size of non nans
        av_frac = (xr.where(da>thresh, 1, 0)).sum(['lat','lon']) / (xr.where(np.isnan(da), 0, 1)).sum(['lat','lon'])
        av_frac = av_frac - 0.5
    return av_frac


def PAVdiff(X_gdd, X_rcm):
    """
    Calculate the difference between the global driving model (gdd) and the regional climate model (rcm).
    :param X_gdd: xarray containing the global driving data
    :param X_rcm: xarray containing the regional climate model
    :return:      xarray containting their difference for each grid-point
    """

    with xr.set_options(keep_attrs=True):
        pav = X_gdd - X_rcm

    return pav

def PAVdiff_rel(X_gdd, X_rcm):
    """
    Calculate the relative difference between the global driving model (gdd) and the regional climate model (rcm).
    :param X_gdd: xarray containing the global driving data
    :param X_rcm: xarray containing the regional climate model
    :return:      xarray containting their relative difference for each grid-point
    """

    with xr.set_options(keep_attrs=True):
        pav = 1 - X_gdd / X_rcm

    return pav

def PAVcorr(X_gdd, X_rcm):
    """
    Calculate difference in terms of spatial correlation between the driving model (gdd) and the regional climate model (rcm).

    :param X_gdd: xarray containing the global driving data
    :param X_rcm: xarray containing the regional climate model
    :return:      xarray containting the AV (corr) for each grid-point
    """
    with xr.set_options(keep_attrs=True):
        out = (1-xr.corr(X_gdd, X_rcm, dim=['lat','lon']))
    return out

def get_latlongrid_xr(ds):
    """
    Get lan/lon grid from input dataset.

    :param ds:     Input xarray from which to extract grid.
    :return:       xarray dataset of lat/lon grid information.
    """
    accept_lon_names = ["lon", "longitude"]
    accept_lat_names = ["lat", "latitude"]
    lon_name = find_dimname_in_acceptable(ds, accept=accept_lon_names)
    lat_name = find_dimname_in_acceptable(ds, accept=accept_lat_names)
    lat   = ds[lat_name].values
    lon   = ds[lon_name].values
    lat_b = guess_bounds(lat)
    lon_b = guess_bounds(lon)
    grid  = xr.Dataset({'longitude': lon,'latitude': lat, 'longitude_b': lon_b, 'latitude_b': lat_b})
    grid['longitude'].attrs = ds[lon_name].attrs
    grid['latitude'].attrs = ds[lat_name].attrs
    # Add mask and ignore the time dimension
    if "time" in ds.dims:
        ds_no_time = ds.isel(time=0)
    else:
        ds_no_time = ds
    mask = xr.where(~np.isnan(ds_no_time), 1, 0)
    mask = mask.rename({lat_name:"latitude", lon_name:"longitude"})
    grid["mask"] = mask
    return grid


def get_resolution(ds, dim):
    """
    Return the resolution of a coordinate.

    :param ds:   Input xarray dataset.
    :param dim:  Coordinate to get resolution for.
    :return:     Resolution
    """
    return ds[dim][1].values - ds[dim][0].values


def regrid_helper(ds, other, exclude_list=["time_bnds"], **kwargs):
    """
    Helper for XESMF regridding.

    :param ds:     Input xarray dataset to be regridded.
    :param other:  Input xarray dataset with reference grid.
    :return:       xarray dataset regridded to other.
    """
    grid_in   = get_latlongrid_xr(ds)
    grid_out  = get_latlongrid_xr(other)
    logger.debug("Grid in:")
    logger.debug(grid_in)
    logger.debug("Grid out:")
    logger.debug(grid_out)
    regridder = xe.Regridder(grid_in, grid_out, **kwargs)
    regridder._grid_in  = None # Otherwise there is trouble with dask
    regridder._grid_out = None # Otherwise there is trouble with dask
    ds_regrid = regridder(ds, keep_attrs=True)
    ds_regrid = rename_helper(ds_regrid, **{"latitude":"lat", "longitude": "lon"}) #< Make sure we get lat and lon back as names and not latitude, longitude
    return ds_regrid


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Use lib_standard instead for now
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def regrid(ds, other, **kwargs):
    """
    Helper for XESMF regridding.

    :param ds:     Input xarray dataset to be regridded.
    :param other:  Input xarray dataset with reference grid.
    :return:       xarray dataset regridded to other.
    """
    if isinstance(ds, list) and isinstance(other, list):
        assert len(ds) == len(other), f"In regrid: ds and other are both lists but have unequal lengths {len(ds)} != {len(other)}"
        for i in range(len(ds)):
            ds[i] = _regrid(ds[i], other[i], **kwargs)
    elif isinstance(ds, list):
        for i in range(len(ds)):
            ds[i] = _regrid(ds[i], other, **kwargs)
    elif isinstance(other, list):
        warnings.warn("Warning: Cannot re-grid dataset to multiple others (i.e. other is a list)\nRe-gridding to the first entry of other!")
        ds = _regrid(ds, other[0], **kwargs)
    else:
        ds = _regrid(ds, other, **kwargs)
    return ds
# def regrid(ds, other, **kwargs):
#     try:
#         ds = ds.to_dataset()
#     except:
#         pass
#     try:
#         other = other.to_dataset()
#     except:
#         pass
#     res = lib_standards.regrid(ds, other)
#     try:
#         res = res.to_dataarray()
#     except:
#         pass
#     return res


def _regrid(ds, other, **kwargs):
    """
    Helper for XESMF regridding.

    :param ds:     Input xarray dataset to be regridded.
    :param other:  Input xarray dataset with reference grid.
    :return:       xarray dataset regridded to other.
    """

    accept_lon_names = ["lon", "longitude"]
    accept_lat_names = ["lat", "latitude"]
    lon_name = find_dimname_in_acceptable(ds, accept=accept_lon_names)
    lat_name = find_dimname_in_acceptable(ds, accept=accept_lat_names)
    lon_name_other = find_dimname_in_acceptable(other, accept=accept_lon_names)
    lat_name_other = find_dimname_in_acceptable(other, accept=accept_lat_names)

    if not "regrid_method" in kwargs:
        #< Check resolution for default regridding method
        dx       = get_resolution(ds, lon_name)
        dx_other = get_resolution(other, lon_name_other)
        dy       = get_resolution(ds, lat_name)
        dy_other = get_resolution(other, lat_name_other)
        if dx > dx_other or dy > dy_other:
            default_method = "bilinear"
        elif dx < dx_other or dy < dy_other:
            default_method = "conservative_normed"
        elif ds[lon_name][0] != other[lon_name_other][0] or ds[lat_name][0] != other[lat_name_other][0]:
            default_method = "bilinear"
        else:
            logger.info("Input dataset and other grid are already identical!")
            return ds
        method = default_method
    else:
        method = kwargs["regrid_method"]
    logger.info(f"Using {method} for re-gridding!")

    if "regrid_dir" in kwargs:
        regrid_dir = kwargs["regrid_dir"] + '/'
    else:
        regrid_dir = f"{tempfile.TemporaryDirectory().name}/"

    if regrid_dir and not os.path.exists(regrid_dir):
        os.makedirs(regrid_dir, exist_ok=True)
        logger.info(f"Created new directory: {regrid_dir}")

    default_weight_filename = f'{regrid_dir}weights_{method}_{ds[lat_name][0].values}_{ds[lon_name][0].values}_{len(ds[lat_name])}x{len(ds[lon_name])}_{other[lat_name_other][0].values}_{other[lon_name_other][0].values}_{len(other[lat_name_other])}x{len(other[lon_name_other])}.nc'
    if not "weight_filename" in kwargs:
        kwargs["weight_filename"] = default_weight_filename

    reuse_weights = False
    if "reuse_regrid_weights" in kwargs:
        if kwargs["reuse_regrid_weights"]:
            reuse_weights = True if os.path.isfile(kwargs["weight_filename"]) else False

    if "reuse_regrid_weights" in kwargs:
        if kwargs["reuse_regrid_weights"]:
            if not os.path.isfile(kwargs["weight_filename"]):
                logger.info(f"Creating weight file: {kwargs['weight_filename']}")
            else:
                logger.info(f"Re-using weight file: {kwargs['weight_filename']}")

    #< Do the regridding
    return regrid_helper(ds, other, method=method, reuse_weights=reuse_weights, filename=kwargs["weight_filename"])


def find_varname_in_acceptable(ds, accept=[]):
    """Given a list of acceptable names find if one of them is in the dataset and return them.

    Args:
        ds (xarray dataset): Input dataset to delete attribute from.
        accept (list): List of acceptable names.

    Returns:
        string : The acceptable name found.
    """
    found_names = list(set(accept) & set(list(ds.keys()))) #< Find intersect between acceptable names and names in dataset
    assert len(found_names) < 2, f"Found more than one name that match accepatable names!\nNames in dataset: {list(ds.keys())}\nAcceptable names: {accept}"
    assert len(found_names) != 0, f"Found no names that match accepatable names!\nNames in dataset: {list(ds.keys())}\nAcceptable names: {accept}"
    return found_names[0]

def no_process(ds):
    return ds

def find_dimname_in_acceptable(ds, accept=[]):
    """Given a list of acceptable names find if one of them is in the dataset and return them.

    Args:
        ds (xarray dataset): Input dataset to delete attribute from.
        accept (list): List of acceptable names.

    Returns:
        string : The acceptable name found.
    """
    found_names = list(set(accept) & set(list(ds.dims))) #< Find intersect between acceptable names and names in dataset
    assert len(found_names) < 2, f"Found more than one name that match accepatable names!\nNames in dataset: {list(ds.keys())}\nAcceptable names: {accept}"
    assert len(found_names) != 0, f"Found no names that match accepatable names!\nNames in dataset: {list(ds.keys())}\nAcceptable names: {accept}"
    return found_names[0]


def makedir(file):
    """Check if directory for the given input file exists, if not create it.

    Args:
        file (string): Input file name

    Returns:
        None
    """
    dir = os.path.dirname(file)
    if dir and not os.path.exists(dir):
        logger.info(f"Create new dir at: {dir}")
        os.makedirs(dir, exist_ok=True)


def write2nc(ds, *args, inlogs=None, **kwargs):
    #< Create directory if it does not exist
    makedir(args[0])
    #< Add new history
    log = cmdprov.new_log(infile_logs=inlogs) # Get new history
    ds.attrs['history'] = log
    #< Save output
    return ds.to_netcdf(*args, **kwargs)


def quantile(da, quantile=None):
    """Calculate quantile for dataarray

    Args:
        da (xarray dataarray): Input dataarray
        quantile (float): Quantile to calculate

    Returns:
        xarray datarray
    """
    #< Re-chunk the data because quantiles cannot be calculated over chunked dimensions
    logger.info(f"Re-chunking data")
    if 'lat' in da.dims:
        da = da.chunk({"time":-1, "lat":"auto"})
    else:
        da = da.chunk({"time":-1})
    #< Calculate quantile
    logger.info(f"Calculating {quantile*100}th quantile")
    X = da.quantile(quantile,"time", skipna=True).load()
    logger.debug(X)
    logger.debug("---------------------------------------------")
    return X


def mean(da):
    """Calculate mean for dataarray

    Args:
        da (xarray dataarray): Input dataarray

    Returns:
        xarray datarray
    """
    #< Calculate mean
    logger.info(f"Calculating mean")
    X = da.mean("time").load()
    logger.debug(X)
    logger.debug("---------------------------------------------")
    return X


def count_above_threshold(da, threshold=None):
    """Calculate the number of times a threshold is exceeded.

    Args:
        da (xarray dataarray): Input dataarray
        threshold (float): Threshold to exceed

    Returns:
        xarray datarray
    """
    #< Calculate quantile
    logger.info(f"Calculating how often {threshold} is exceeded")
    X = xr.where(da>threshold, 1, 0).sum("time").load()
    return X


def count_below_threshold(da, threshold=None):
    """Calculate the number of times a a value is below the threshold.

    Args:
        da (xarray dataarray): Input dataarray
        threshold (float): Threshold to exceed

    Returns:
        xarray datarray
    """
    #< Calculate quantile
    logger.info(f"Calculating how often below {threshold}")
    X = xr.where(da<threshold, 1, 0).sum("time").load()
    return X


def heatmap(arr, xdim="", ydim="", xlabels='', ylabels='', fmt="{:.2f}", title='', cmap=None, cmap_centre=None, vmin=None, vmax=None, robust=False):

    #< Make sure xdim and ydim are in dataset
    if not xdim in arr.dims:
        if xdim in arr.coords:
            arr = arr.expand_dims(xdim)
        else:
            exit(1)
    if not ydim in arr.dims:
        if ydim in arr.coords:
            arr = arr.expand_dims(ydim)
        else:
            exit(1)

    #< Calculate a mean over all other dimensions (not xdim and ydim)
    meandims = []
    for d in arr.dims:
        if not d==xdim and not d==ydim:
            meandims.append(d)
    arr = arr.mean(meandims)

    #< Correctly order x and y dimension
    arr = arr.transpose(*[ydim,xdim])

    #< Get the data
    arr_data = arr.load().data

    if cmap:
        try:
            cmap = getattr(cmaps, cmap)
        except:
            cmap = cmap

    if robust:
        vmin=arr.quantile(0.1); vmax=arr.quantile(0.9)

    #< Adjust the colorbar values
    if cmap_centre != None:
        divnorm = colors.TwoSlopeNorm(vcenter=cmap_centre)
    else:
        divnorm = None

    #< Create the figure
    fig, ax = plt.subplots()
    im = ax.imshow(arr_data, cmap=cmap, norm=divnorm, vmin=vmin, vmax=vmax)
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(arr[xdim].values)))
    ax.set_yticks(np.arange(len(arr[ydim].values)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(xlabels if xlabels else arr[xdim].values)
    ax.set_yticklabels(ylabels if ylabels else arr[ydim].values)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(arr[ydim].values)):
        for j in range(len(arr[xdim].values)):
            text = ax.text(j, i, fmt.format(arr_data[i, j]),
                           ha="center", va="center", color="k", fontsize=15)
    # Set title
    if title:
        ax.set_title(title)

    plt.tight_layout()

    return fig


def tidy_filename(filename):
    #< Replace double underscore with single underscore
    while "__" in filename:
        filename = filename.replace("__", "_")
    if os.path.basename(filename).startswith("_"):
        filename = os.path.join(os.path.dirname(filename), os.path.basename(filename)[1:])
    return filename


def get_filename(measure, variable, gcm, scenario, rcm, obs, freq, region, season, datestart_hist="", dateend_hist="", datestart_fut="", dateend_fut="", upscale2ref=False, upscale2gcm=False, kwargs=dict()):
    if (upscale2gcm and upscale2ref):
        raise Exception(f"upscale2gcm and upscale2ref cannot both be True!")
    if upscale2ref:
        basedir = "/g/data/tp28/Climate_Hazards/ACS_added_value/added_value_upscale_ref"
    elif upscale2gcm:
        basedir = "/g/data/tp28/Climate_Hazards/ACS_added_value/added_value_upscale_gcm"
    else:
        basedir = "/g/data/tp28/Climate_Hazards/ACS_added_value/added_value_rcm_grid"
    # Create a list of kwargs in the style of [key0, val0, key1, val1, ... keyN, valN]
    kwargs_list = []
    for key in kwargs:
        kwargs_list.append(key)
        kwargs_list.append(kwargs[key])
    ofile = f"{basedir}/{measure}_{variable}_{'_'.join([str(i).replace('.','p') for i in kwargs_list])}_{gcm}_{scenario}_{rcm}_{obs}_{freq}_{region.replace(' ', '_')}_{season}"
    if upscale2gcm:
        ofile = ofile + "_upscale2gcm_"
    if upscale2ref:
        ofile = ofile + "_upscale2ref_"
    if datestart_hist and dateend_hist:
        ofile = ofile + f"_{datestart_hist}-{dateend_hist}"
    if datestart_fut and dateend_fut:
        ofile = ofile + f"_{datestart_fut}-{dateend_fut}"
    ofile = ofile + ".nc"
    ofile = tidy_filename(ofile)
    return ofile


def get_files(measure, variable="*", gcm="*", scen="*", region="*", season="*", rcm="BARPA-R", freq="day", upscale2ref=False, upscale2gcm=False, kwargs=dict()):
    kwargs_in = kwargs.copy()
    kwargs = {}
    if measure.startswith("AV"):
        kwargs["datestart_hist"] = "19850101"
        kwargs["dateend_hist"] = "20141231"
        obs = "AGCD"
    elif measure.startswith("PAV") or measure.startswith("RAV"):
        kwargs["datestart_hist"] = "19850101"
        kwargs["dateend_hist"] = "20141231"
        kwargs["datestart_fut"] = "20700101"
        kwargs["dateend_fut"] = "20991231"
        if measure.startswith("RAV"):
            obs = "AGCD"
        else:
            obs = ""
    kwargs["kwargs"] = kwargs_in

    filename = get_filename(
        measure,
        variable,
        gcm,
        scen,
        rcm,
        obs,
        freq,
        region,
        season,
        **kwargs,
        upscale2ref=upscale2ref,
        upscale2gcm=upscale2gcm,
    )
    file_list = glob.glob(filename)
    if not file_list:
        raise Exception(f"No files found! {filename}")
    return file_list


def match_pattern(filename, addon="_upscale2ref_"):
                  # AVse                    _pr                        _quantile                          _0p9                        _ACCESS-CM2            _historical             _BARPA-R               _AGCD                 _day                     _Australia                 _MAM                                         _upscale2ref_19850101-20141231.nc
    pattern1 = f"(?P<measure>[-a-zA-Z0-9]+)_(?P<varname>[-a-zA-Z0-9]+)_(?P<quantile_method>[-a-zA-Z0-9]+)_(?P<quantile>[-a-zA-Z0-9]+)_(?P<gcm>[-a-zA-Z0-9]+)_(?P<scen>[-a-zA-Z0-9]+)_(?P<rcm>[-a-zA-Z0-9]+)_(?P<obs>[-a-zA-Z0-9]+)_(?P<freq>[-a-zA-Z0-9]+)_(?P<region>[-_a-zA-Z0-9]+)_(?P<season>(annual)|(DJF)|(MAM)|(JJA)|(SON)){addon}(?P<datestart1>[0-9]+)-(?P<dateend1>[0-9]+).nc"
    pattern2 = f"(?P<measure>[-a-zA-Z0-9]+)_(?P<varname>[-a-zA-Z0-9]+)_(?P<quantile_method>[-a-zA-Z0-9]+)_(?P<quantile>[-a-zA-Z0-9]+)_(?P<gcm>[-a-zA-Z0-9]+)_(?P<scen>[-a-zA-Z0-9]+)_(?P<rcm>[-a-zA-Z0-9]+)_(?P<obs>[-a-zA-Z0-9]+)_(?P<freq>[-a-zA-Z0-9]+)_(?P<region>[-_a-zA-Z0-9]+)_(?P<season>(annual)|(DJF)|(MAM)|(JJA)|(SON)){addon}(?P<datestart1>[0-9]+)-(?P<dateend1>[0-9]+)_(?P<datestart2>[0-9]+)-(?P<dateend2>[0-9]+).nc"
    pattern3 = f"(?P<measure>[-a-zA-Z0-9]+)_(?P<varname>[-a-zA-Z0-9]+)_(?P<quantile_method>[-a-zA-Z0-9]+)_(?P<quantile>[-a-zA-Z0-9]+)_(?P<gcm>[-a-zA-Z0-9]+)_(?P<scen>[-a-zA-Z0-9]+)_(?P<rcm>[-a-zA-Z0-9]+)_(?P<freq>[-a-zA-Z0-9]+)_(?P<region>[-_a-zA-Z0-9]+)_(?P<season>(annual)|(DJF)|(MAM)|(JJA)|(SON)){addon}(?P<datestart1>[0-9]+)-(?P<dateend1>[0-9]+)_(?P<datestart2>[0-9]+)-(?P<dateend2>[0-9]+).nc"
    dirname = os.path.dirname(filename)
    basename = os.path.basename(filename)
    if re.search(pattern1, basename):
        matches = re.search(pattern1, basename).groupdict()
    elif re.search(pattern2, basename):
        matches = re.search(pattern2, basename).groupdict()
    elif re.search(pattern3, basename):
        matches = re.search(pattern3, basename).groupdict()
    return matches

def get_varname_from_file(filename, addon="_upscale2ref_"):
    return match_pattern(filename, addon)["varname"]

def get_gcm_from_file(filename, addon="_upscale2ref_"):
    return match_pattern(filename, addon)["gcm"]

def get_scen_from_file(filename, addon="_upscale2ref_"):
    return match_pattern(filename, addon)["scen"]

def get_region_from_file(filename, addon="_upscale2ref_"):
    return match_pattern(filename, addon)["region"]

def get_season_from_file(filename, addon="_upscale2ref_"):
    return match_pattern(filename, addon)["season"]

def get_rcm_from_file(filename, addon="_upscale2ref_"):
    return match_pattern(filename, addon)["rcm"]

def get_quantile_from_file(filename, addon="_upscale2ref_"):
    return match_pattern(filename, addon)["quantile"]


def load_av_data(measure, variable="*", gcm="*", scen="*", region="*", season="*", rcm="BARPA-R", freq="day", kwargs=dict(), upscale2ref=False, upscale2gcm=False, region_mask=None):
    if upscale2ref:
        addon = "_upscale2ref_"
    elif upscale2gcm:
        addon = "_upscale2gcm_"
    else:
        addon = "_"

    file_list = sorted(get_files(
        measure,
        variable,
        gcm,
        scen,
        region,
        season,
        rcm,
        freq,
        upscale2ref,
        upscale2gcm,
        kwargs,
    ))
    varnames = []
    gcms = []
    scens = []
    regions = []
    seasons = []
    rcms = []
    quantiles = []
    for f in file_list:
        if not get_varname_from_file(f, addon) in varnames:
            varnames.append(get_varname_from_file(f, addon))
        if not get_gcm_from_file(f, addon) in gcms:
            gcms.append(get_gcm_from_file(f, addon))
        if not get_scen_from_file(f, addon) in scens:
            scens.append(get_scen_from_file(f, addon))
        if not get_region_from_file(f, addon) in regions:
            regions.append(get_region_from_file(f, addon))
        if not get_season_from_file(f, addon) in seasons:
            seasons.append(get_season_from_file(f, addon))
        if not get_rcm_from_file(f, addon) in rcms:
            rcms.append(get_rcm_from_file(f, addon))
        if not get_quantile_from_file(f, addon) in quantiles:
            quantiles.append(get_quantile_from_file(f, addon))

    ds = []
    for varname in varnames:
        ds_gcms = []
        for gcm in gcms:
            ds_scens = []
            for scen in scens:
                ds_regions = []
                for region in regions:
                    ds_seasons = []
                    for season in seasons:
                        ds_rcms = []
                        for rcm in rcms:
                            ds_quantiles = []
                            for quantile in quantiles:
                                kwargs = dict(quantile=quantile)
                                try:
                                    one_file_list = get_files(
                                        measure,
                                        varname,
                                        gcm,
                                        scen,
                                        region,
                                        season,
                                        rcm,
                                        freq,
                                        upscale2ref,
                                        upscale2gcm,
                                        kwargs,
                                    )
                                    _ds = xr.open_dataset(one_file_list[0])
                                except Exception as e:
                                    _ds = xr.zeros_like(_ds)
                                    _ds = xr.where(_ds == 0, np.nan, np.nan)
                                ds_quantiles.append( _ds )
                            ds_rcms.append( xr.concat(ds_quantiles, pd.Index(quantiles, name="quantile")) )
                        ds_seasons.append( xr.concat(ds_rcms, pd.Index(rcms, name="rcm")) )
                    ds_regions.append( xr.concat(ds_seasons, pd.Index(seasons, name="season")) )
                ds_scens.append( xr.concat(ds_regions, pd.Index(regions, name="region")) )
            ds_gcms.append( xr.concat(ds_scens, pd.Index(scens, name="scenario")) )
        ds.append( xr.concat(ds_gcms, pd.Index(gcms, name="gcm")) )
    ds = xr.concat(ds, pd.Index(varnames, name="variable"))

    # Make sure we sort this nicely
    ds = ds.sortby(ds["quantile"])
    ds = ds.sortby(ds["rcm"])
    ds = ds.sortby(ds["season"])
    ds = ds.sortby(ds["region"])
    ds = ds.sortby(ds["scenario"])
    ds = ds.sortby(ds["gcm"])
    ds = ds.sortby(ds["variable"])

    # Maybe mask
    if region_mask:
        ds = lib_spatial.apply_region_mask(ds, region_mask)

    return ds


def av_da2dict(da):
    if len(da.dims) != 2:
        raise Exception(f"Can only convert 2-dimensional datasets to dictionary!")
    data_dict = {}
    dim0 = list(da.dims)[0]
    dim1 = list(da.dims)[1]
    for d0 in da[dim0].data:
        for d1 in da[dim1].data:
            if not d0 in data_dict:
                data_dict[d0] = {}
            data_dict[d0][d1] = da.sel(**{f"{dim0}":d0, f"{dim1}":d1}).values
    return data_dict


def mean_over_other(ds, *exclude_dims):
    all_dims = ds.dims
    mean_dims = []
    for dim in all_dims:
        if not dim in exclude_dims:
            mean_dims.append(dim)
    print(f"Calculating mean over {mean_dims}")
    return ds.mean(mean_dims)


def hinton_wrapper(da, xdim, ydim, sig=None, add_ens_avfrac_mean=False, ens_mean_dim=None, robust=False):
    if add_ens_avfrac_mean:
        ens_mean = frac_ss(da)
        if ens_mean_dim is None or not ens_mean_dim in ens_mean.dims:
            raise ValueError(f"{ens_mean_dim} dimension not found! Cannot calculate ensemble mean!")
        ens_mean = ens_mean.mean(ens_mean_dim)
        ens_mean = ens_mean.expand_dims(dim={ens_mean_dim: ["Ensemble-mean"]})
    da_mean = mean_over_other(da.squeeze(), xdim, ydim)
    if add_ens_avfrac_mean:
        da_mean = xr.concat([da_mean, ens_mean], dim=ens_mean_dim)
        sig = xr.concat([sig, ens_mean], dim=ens_mean_dim)
        sig = xr.where(sig[ens_mean_dim]=="Ensemble-mean", None, sig)

    da_mean = da_mean.transpose(xdim,ydim)
    sig = sig.transpose(xdim,ydim)
    data_dict = av_da2dict(da_mean)
    sig_dict = None
    sig_dict_bool = None
    if sig is not None:
        sig = sig.squeeze()
        sig_dict = av_da2dict(sig)
        sig_dict_bool = copy.deepcopy(sig_dict)
        for key in data_dict:
            for key2 in data_dict[key]:
                # if key == "Ensemble-mean" or key2 == "Ensemble-mean":
                #     if not "Ensemble-mean" in sig_dict_bool:
                #         sig_dict_bool[key] = {}
                #     sig_dict_bool[key][key2] = None
                if data_dict[key][key2] > 0. and sig_dict[key][key2] > 0.:
                    sig_dict_bool[key][key2] = 1
                elif data_dict[key][key2] < 0. and sig_dict[key][key2] < 0.:
                    sig_dict_bool[key][key2] = 1
                else:
                    sig_dict_bool[key][key2] = None
                # print(key, key2, data_dict[key][key2], sig_dict[key][key2], sig_dict_bool[key][key2])
    fig, ax = plt.subplots()
    lib_standards.hinton(ax, data_dict, sig=sig_dict_bool, robust=robust)
    return ax


def table_plot_wrapper(da, xdim, ydim, sig=None, add_ens_avfrac_mean=False, robust=False):
    if add_ens_avfrac_mean:
        ens_mean = frac_ss(da)
        if not "gcm" in ens_mean.dims:
            raise ValueError("gcm dimension not found! Cannot calculate ensemble mean!")
        if not "rcm" in ens_mean.dims:
            raise ValueError("rcm dimension not found! Cannot calculate ensemble mean!")
        ens_mean = ens_mean.mean(["gcm", "rcm"])
        ens_mean = ens_mean.expand_dims(dim={"gcm": ["Ensemble-mean"]})
    da_mean = mean_over_other(da.squeeze(), xdim, ydim)
    if add_ens_avfrac_mean:
        da_mean = xr.concat([da_mean, ens_mean], dim="gcm")

    da_mean = da_mean.transpose(xdim,ydim)
    data_dict = av_da2dict(da_mean)
    sig_dict = None
    sig_dict_bool = None
    if sig is not None:
        sig = sig.squeeze()
        sig_dict = av_da2dict(sig)
        sig_dict_bool = copy.deepcopy(sig_dict)
        for key in data_dict:
            for key2 in data_dict[key]:
                if key == "Ensemble-mean":
                    if not "Ensemble-mean" in sig_dict_bool:
                        sig_dict_bool[key] = {}
                    sig_dict_bool[key][key2] = None
                elif data_dict[key][key2] > 0. and sig_dict[key][key2] > 0.:
                    sig_dict_bool[key][key2] = 1
                elif data_dict[key][key2] < 0. and sig_dict[key][key2] < 0.:
                    sig_dict_bool[key][key2] = 1
                else:
                    sig_dict_bool[key][key2] = None
                # print(key, key2, data_dict[key][key2], sig_dict[key][key2], sig_dict_bool[key][key2])
    fig, ax = plt.subplots()
    lib_standards.table_plot(ax, data_dict)
    return ax


def plot_map(da, **kwargs):
    mean_over_other(da, "lat", "lon").plot.pcolormesh(subplot_kws=dict(projection=ccrs.PlateCarree()), transform=ccrs.PlateCarree(), **kwargs)
    ax = plt.gca()
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.coastlines()
    return ax