import xarray as xr
import numpy as np
import os
import dask
import datetime
import pandas as pd
import xesmf as xe
import matplotlib.pyplot as plt
import warnings


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
        print(f'No files for {ifiles}')
    ds = xr.open_mfdataset(ifiles, **read_kwargs)
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
            print("Input dataset and other grid are already identical!")
            return ds
        method = default_method
    else:
        method = kwargs["regrid_method"]

    if "regrid_dir" in kwargs:
        regrid_dir = kwargs["regrid_dir"] + '/'
    else:
        regrid_dir = ""

    if regrid_dir and not os.path.exists(regrid_dir):
        os.makedirs(regrid_dir, exist_ok=True)
        print(f"Created new directory: {regrid_dir}")

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
                print(f"Creating weight file: {kwargs['weight_filename']}")
            else:
                print(f"Re-using weight file: {kwargs['weight_filename']}")

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