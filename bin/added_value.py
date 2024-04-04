import numpy as np
import xarray as xr
import pandas as pd
import dask
import netCDF4
import h5py
import tempfile
import argparse
import cmdline_provenance as cmdprov
import matplotlib.pyplot as plt
import time
import os
import dask.distributed
import sys
import warnings
import lib
import json
import lib_spatial
import logging


#< Get logger
logger = lib.get_logger(__name__)


def parse_arguments():
    # User argument input
    parser = argparse.ArgumentParser(description='Script for calculating added value using GCM, RCM and observations as input.')
    parser.add_argument("--ifiles-gcm", dest='ifiles_gcm', nargs='*', type=str, default=[], help="Input GCM files")
    parser.add_argument("--ifiles-rcm", dest='ifiles_rcm', nargs='*', type=str, default=[], help="Input RCM files")
    parser.add_argument("--ifiles-obs", dest='ifiles_obs', nargs='*', type=str, default=[], help="Input reference files")

    parser.add_argument("--varname-gcm", dest='varname_gcm', nargs='?', type=str, default="", help="Variable name in GCM files")
    parser.add_argument("--varname-rcm", dest='varname_rcm', nargs='?', type=str, default="", help="Variable name in RCM files")
    parser.add_argument("--varname-obs", dest='varname_obs', nargs='?', type=str, default="", help="Variable name in reference files")

    parser.add_argument("--region", dest='region', nargs='?', type=str, default="", help="Region masking using lib_spatial.py")

    parser.add_argument("--agcd_mask", dest='agcd_mask', action='store_true', help='Whether to apply masking for AGCD precipitation data')

    parser.add_argument("--process", dest='process', nargs='?', type=str, default="", help="Process to get added value for (e.g., quantile)")
    parser.add_argument("--process-kwargs", dest='process_kwargs', nargs='?', type=json.loads, default="{}", help="Kwargs to pass to process function (e.g., \'{\"quantile\": 0.95}\' 0.95 for quantile)")
    parser.add_argument("--distance-measure", dest='distance_measure', nargs='?', type=str, default="", help="Distance measure to use for AV calculation")

    parser.add_argument("--datestart", dest='datestart', nargs='?', type=str, default="", help="Start date of analysis period")
    parser.add_argument("--dateend", dest='dateend', nargs='?', type=str, default="", help="End date of analysis period")
    parser.add_argument("--months", dest='months', nargs='*', type=int, default=[], help="Select only certain months (e.g. 12 1 2 for DJF")

    parser.add_argument("--ofile", dest='ofile', nargs='?', type=str, default="av.nc", help="Path and name of output file")

    parser.add_argument("--lat0", dest='lat0', nargs='?', type=float, default=-999, help="Lower latitude to select")
    parser.add_argument("--lat1", dest='lat1', nargs='?', type=float, default=-999, help="Upper latitude to select")
    parser.add_argument("--lon0", dest='lon0', nargs='?', type=float, default=-999, help="Lower longitude to select")
    parser.add_argument("--lon1", dest='lon1', nargs='?', type=float, default=-999, help="Upper longitude to select")

    parser.add_argument("--stations", dest='stations', default=False, action='store_true',help="Obs data is point-based station data")

    parser.add_argument("--reuse-X", dest='reuse_X', action='store_true', help="Reuse the regridded climate statistics")
    parser.add_argument("--ifile-X-gcm", nargs='?', type=str, default=None, help="Input statistic GCM file")
    parser.add_argument("--ifile-X-rcm", nargs='?', type=str, default=None, help="Input statistic RCM file")
    parser.add_argument("--ifile-X-obs", nargs='?', type=str, default=None, help="Input statistic reference file")

    parser.add_argument("--nthreads", dest='nthreads', nargs='?', type=int, const='', default=1, help="Number of threads.")
    parser.add_argument("--nworkers", dest='nworkers', nargs='?', type=int, const='', default=2, help="Number of workers.")

    parser.add_argument("--log-level", dest='loglevel', nargs='?', type=str, default="INFO", help="Amount of log output")

    return parser



def added_value(da_gcm, da_rcm, da_obs, process, process_kwargs={}, distance_measure="AVrmse", region=None, reuse_X=False, agcd_mask=False, stations=False,
                ifile_X_rcm=None, ifile_X_gcm=None, ifile_X_obs=None):
    """Calculate added value statistic from driving model (da_gcm), regional model (da_rcm) and reference (da_obs) dataarray

    Args:
        da_gcm (xarray dataarray): Driving model data
        da_rcm (xarray dataarray): Regional model data
        da_obs (xarray dataarray): Reference data
        process (str): Process to calculate AV for (e.g., quantile)
        process_kwargs (dict): Kwargs to pass to "process" (e.g., {'quantile':0.95})
        measure (str): Distance measure to use for added value calculation
        region (str): Region passed to lib_spatial.py for masking
        write_X (bool): Should the regridded climate statistic be written out too?
        agcd_mask (bool): Apply AGCD mask based on AGCD precip data
        stations (bool): Is reference data a point-based dataset?

    Returns:
        xarray dataset : Added value
    """
    #< Make sure all dataarrays have the same units
    assert "units" in da_gcm.attrs, f"da_gcm has no units attribute"
    assert "units" in da_rcm.attrs, f"da_rcm has no units attribute"
    assert "units" in da_obs.attrs, f"da_obs has no units attribute"
    assert da_gcm.attrs["units"] == da_rcm.attrs["units"] == da_obs.attrs["units"], f"Not all dataarrays have the same units: {da_gcm.attrs['units']} != {da_rcm.attrs['units']} != {da_obs.attrs['units']}"
    #< Search for "process" function in library and run it on all three dataarrays
    if hasattr(lib, process):
        fun = getattr(lib, process)
    else:
        assert False, f"{process} not implemented!"


    if reuse_X and os.path.isfile(ifile_X_rcm):
        logger.info(f"Using existing {ifile_X_rcm} for RCM ")
        X_rcm = xr.open_dataarray(ifile_X_rcm)
        logger.debug(X_rcm)
    elif not reuse_X or not os.path.isfile(ifile_X_rcm):
        logger.info(f"Calculating {process} for RCM")
        X_rcm = fun(da_rcm, **process_kwargs)
        if not stations:
            #< Mask data
            if not region is None:
                logger.info("Masking X_rcm.")
                X_rcm = lib_spatial.apply_region_mask(X_rcm, region.replace("_", " "))
                logger.debug(X_rcm)
            #< AGCD mask
            if agcd_mask:
                logger.info("Masking X_rcm with AGCD mask")
                X_rcm = lib_spatial.apply_agcd_data_mask(X_rcm)
                logger.debug(X_rcm)
    if reuse_X and not os.path.isfile(ifile_X_rcm):
        logger.info(f"Saving {ifile_X_rcm} for RCM to netcdf")
        lib.write2nc(X_rcm, ifile_X_rcm)
    

    if reuse_X and os.path.isfile(ifile_X_gcm):
        logger.info(f"Using existing {ifile_X_gcm} for GCM ")
        X_gcm = xr.open_dataarray(ifile_X_gcm)
        logger.debug(X_gcm)
    elif not reuse_X or not os.path.isfile(ifile_X_gcm):
        logger.info(f"Calculating {process} for GCM")
        X_gcm = fun(da_gcm, **process_kwargs)
        if not stations:
            logger.info(f"Regridding GCM data to RCM grid")
            X_gcm = lib.regrid(X_gcm, X_rcm)
            #< Mask data
            if not region is None:
                logger.info("Masking X_gcm.")
                X_gcm = lib_spatial.apply_region_mask(X_gcm, region.replace("_", " "))
                logger.debug(X_gcm)
            #< AGCD mask
            if agcd_mask:
                logger.info("Masking X_gcm with AGCD mask")
                X_gcm = lib_spatial.apply_agcd_data_mask(X_gcm)
                logger.debug(X_gcm)
    if reuse_X and not os.path.isfile(ifile_X_gcm):
        logger.info(f"Saving {ifile_X_gcm} for GCM to netcdf")
        lib.write2nc(X_gcm, ifile_X_gcm)


    if reuse_X and os.path.isfile(ifile_X_obs):
        logger.info(f"Using existing {ifile_X_obs} for OBS ")
        X_obs = xr.open_dataarray(ifile_X_obs)
        logger.debug(X_obs)
    elif not reuse_X or not os.path.isfile(ifile_X_obs):
        logger.info(f"Calculating {process} for OBS")
        X_obs = fun(da_obs, **process_kwargs)
        if not stations:
            logger.info(f"Regridding OBS data to RCM grid")
            X_obs = lib.regrid(X_obs, X_rcm)
            #< Mask data
            if not region is None:
                logger.info("Masking X_obs.")
                X_obs = lib_spatial.apply_region_mask(X_obs, region.replace("_", " "))
                logger.debug(X_obs)
            #< AGCD mask
            if agcd_mask:
                logger.info("Masking X_obs with AGCD mask")
                X_obs = lib_spatial.apply_agcd_data_mask(X_obs)
                logger.debug(X_obs)
    if reuse_X and not os.path.isfile(ifile_X_obs):
        logger.info(f"Saving {ifile_X_obs} for OBS to netcdf")
        lib.write2nc(X_obs, ifile_X_obs)

    #< Calculate added value
    logger.info(f"Calculating added value using {distance_measure}")
    if hasattr(lib, distance_measure):
        fun = getattr(lib, distance_measure)
    else:
        assert False, f"Distance measure of {distance_measure} not implemented!"
    av = fun(X_obs, X_gcm, X_rcm)
    logger.debug(av)
    logger.debug("---------------------------------------------")
    #< Convert av to a dataset
    av = av.to_dataset(name="av")
    #< Return
    return av


def main():
    # Load the logger
    logger.info(f"Start")

    #< Get user arguments
    parser = parse_arguments()
    args = parser.parse_args()

    #< Open datasets
    logger.info(f"Opening datasets")
    ds_gcm = lib.open_dataset(args.ifiles_gcm)
    ds_rcm = lib.open_dataset(args.ifiles_rcm)
    ds_obs = lib.open_dataset(args.ifiles_obs)
    logger.debug(ds_gcm)
    logger.debug(ds_rcm)
    logger.debug(ds_obs)

    #< Get the history of the input files
    inlogs = {}
    if "history" in ds_gcm.attrs:
        inlogs[f"gcm"] = ds_gcm.attrs["history"]
    if "history" in ds_rcm.attrs:
        inlogs[f"rcm"] = ds_rcm.attrs["history"]
    if "history" in ds_obs.attrs:
        inlogs[f"obs"] = ds_obs.attrs["history"]

    #< Fetch variable from dataset
    logger.info(f"Extracting variables from datasets")
    da_gcm = ds_gcm[args.varname_gcm]
    da_rcm = ds_rcm[args.varname_rcm]
    da_obs = ds_obs[args.varname_obs]
    logger.debug(da_gcm)
    logger.debug(da_rcm)
    logger.debug(da_obs)
    logger.debug("---------------------------------------------")

    #< Cut all dataarrays to same time period
    logger.info(f"Selecting time period")
    da_gcm = da_gcm.sel(time=slice(args.datestart, args.dateend))
    da_rcm = da_rcm.sel(time=slice(args.datestart, args.dateend))
    da_obs = da_obs.sel(time=slice(args.datestart, args.dateend))
    logger.debug(da_gcm)
    logger.debug(da_rcm)
    logger.debug(da_obs)
    logger.debug("---------------------------------------------")

    #< Select certain months
    if args.months:
        logger.info(f"Selecting months {args.months}")
        da_gcm = da_gcm.sel(time=da_gcm.time.dt.month.isin(args.months))
        da_rcm = da_rcm.sel(time=da_rcm.time.dt.month.isin(args.months))
        da_obs = da_obs.sel(time=da_obs.time.dt.month.isin(args.months))
        logger.debug(da_gcm)
        logger.debug(da_rcm)
        logger.debug(da_obs)
        logger.debug("---------------------------------------------")

    if args.stations:
        print(args.stations,'stations')
        # extract data from nearest points to stations
        lon = da_obs.lon
        lat = da_obs.lat
        logger.info(f"Selecting station data")
        da_gcm = da_gcm.sel(lat=xr.DataArray(lat, dims='station'), lon=xr.DataArray(lon, dims='station'), method='nearest').load()
        da_rcm = da_rcm.sel(lat=xr.DataArray(lat, dims='station'), lon=xr.DataArray(lon, dims='station'), method='nearest').load()
        da_obs = da_obs.load()
    else:
 
        #< Cut all dataarrays to the same domain
        if args.lat0!=-999 and args.lat1!=-999 and args.lon0!=-999 and args.lon1!=-999:
            logger.info(f"Selecting domain")
            da_gcm = da_gcm.sel(lat=slice(args.lat0, args.lat1), lon=slice(args.lon0, args.lon1))
            da_rcm = da_rcm.sel(lat=slice(args.lat0, args.lat1), lon=slice(args.lon0, args.lon1))
            da_obs = da_obs.sel(lat=slice(args.lat0, args.lat1), lon=slice(args.lon0, args.lon1))
            logger.debug(da_gcm)
            logger.debug(da_rcm)
            logger.debug(da_obs)
            logger.debug("---------------------------------------------")

    #< Calculate added value
    av = added_value(
        da_gcm,
        da_rcm,
        da_obs,
        args.process,
        args.process_kwargs,
        distance_measure=args.distance_measure,
        region=args.region,
        agcd_mask=args.agcd_mask,
        stations=args.stations,
        reuse_X=args.reuse_X,
        ifile_X_rcm=args.ifile_X_rcm,
        ifile_X_gcm=args.ifile_X_gcm,
        ifile_X_obs=args.ifile_X_obs
    )
    logger.debug("Added values looks like:")
    logger.debug(av)

    #< Save added value to netcdf
    logger.info("Saving to netcdf")
    lib.write2nc(av, args.ofile, inlogs=inlogs)

    logger.debug(f"Done")


    


if __name__ == '__main__':

    dask.config.set({
        'array.chunk-size': "1024 MiB",
        'distributed.comm.timeouts.connect': '60s',
        'distributed.comm.timeouts.tcp': '60s',
        'distributed.comm.retry.count': 5,
        'distributed.scheduler.allowed-failures': 10,
        "distributed.scheduler.worker-saturation": 1.1, #< This should use the new behaviour which helps with memory pile up
    })

    parser        = parse_arguments()
    args          = parser.parse_args()
    nthreads     = args.nthreads
    nworkers     = args.nworkers

    try:
        memory_limit = '1000mb' if os.environ["HOSTNAME"].startswith("gadi-login") else int(os.environ["PBS_VMEM"]) / nworkers
    except:
        memory_limit = None
    memory_limit = os.getenv('MEMORY_LIMIT', memory_limit)
    client       = dask.distributed.Client(n_workers = nworkers, threads_per_worker = nthreads,
                                           memory_limit = memory_limit, local_directory = tempfile.mkdtemp(),
                                           silence_logs = logging.ERROR,
                                        ) 

    #< Set the logging level
    logger.setLevel(args.loglevel)

    #< Print client summary
    logger.info('### Client summary')
    logger.info(client)

    #< Call the main function
    main()

    #< Close the client
    client.close()
