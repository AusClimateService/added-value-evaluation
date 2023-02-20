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


def get_logger(name, level='debug'):
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
logger = get_logger(__name__)


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
        
            elif ds[key].attrs["units"] == "kg m-2 s-1":
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
    regridder = xe.Regridder(grid_in, grid_out, **kwargs)
    regridder._grid_in  = None # Otherwise there is trouble with dask
    regridder._grid_out = None # Otherwise there is trouble with dask
    ds_regrid = regridder(ds, keep_attrs=True)
    ds_regrid = rename_helper(ds_regrid, **{"latitude":"lat", "longitude": "lon"}) #< Make sure we get lat and lon back as names and not latitude, longitude
    return ds_regrid


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
        regrid_dir = ""

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
    da = da.chunk({"time":-1, "lat":"auto"})
    #< Calculate quantile
    logger.info(f"Calculating {quantile*100}th quantile")
    X = da.quantile(quantile,"time", skipna=True).load()
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


def iclim_test(da, threshold=None):
    """Calculate the number of times a a value is below the threshold.

    Args:
        da (xarray dataarray): Input dataarray
        threshold (float): Threshold to exceed

    Returns:
        xarray datarray
    """
    
    #< Save da to a file so icclim can work on it
    import uuid
    tmp_file = "tmp_icclim.nc"
    out_file = f"/scratch/tp28/cst565/{uuid.uuid4()}.nc"
    name = da.name
    logger.info(f"Saving dataset to {tmp_file}")
    da.to_netcdf(tmp_file)
    #< Run the file through icclim
    logger.info(f"Running icclim on {tmp_file}")
    import icclim
    icclim.index(index_name="SU", in_files=tmp_file, var_name=name, out_file=out_file)
    #< Open the icclim output file and return
    logger.info(f"Open icclim output {out_file}")
    X = xr.open_dataset(out_file)
    X = X.mean("time").load()
    print(X)

    return X