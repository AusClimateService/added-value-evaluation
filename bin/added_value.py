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

    parser.add_argument("--ifiles-mask", dest='ifiles_mask', nargs='*', type=str, default=[], help="Input masking files")
    parser.add_argument("--varname-mask", dest='varname_mask', nargs='?', type=str, default="", help="Variable name in masking files")
    parser.add_argument("--value-mask", dest='value_mask', nargs='?', type=float, default=-999, help="Value to use for masking (e.g. larger than 0")
    parser.add_argument("--op-mask", dest='op_mask', nargs='?', type=str, default="", help="Operation to use for masking (e.g. larger, smaller")

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

    parser.add_argument("--return-X", dest='return_X', nargs='?', type=lib.str2bool, default="False", help="Also return the regridded climate statistics")

    parser.add_argument("--nthreads", dest='nthreads', nargs='?', type=int, const='', default=1, help="Number of threads.")
    parser.add_argument("--nworkers", dest='nworkers', nargs='?', type=int, const='', default=2, help="Number of workers.")

    parser.add_argument("--log-level", dest='loglevel', nargs='?', type=str, default="INFO", help="Amount of log output")

    return parser



def added_value(da_gcm, da_rcm, da_obs, process, process_kwargs={}, distance_measure="AVrmse", mask=None, return_X=False):
    """Calculate added value statistic from driving model (da_gcm), regional model (da_rcm) and reference (da_obs) dataarray

    Args:
        da_gcm (xarray dataarray): Driving model data
        da_rcm (xarray dataarray): Regional model data
        da_obs (xarray dataarray): Reference data
        process (str): Process to calculate AV for (e.g., quantile)
        process_kwargs (dict): Kwargs to pass to "process" (e.g., {'quantile':0.95})
        measure (str): Distance measure to use for added value calculation
        mask (xarray dataarray): Array (with 0 & 1) used for masking.
        write_X (bool): Should the regridded climate statistic be written out too?

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
    logger.info(f"Calculating {process}")
    X_gcm = fun(da_gcm, **process_kwargs)
    X_rcm = fun(da_rcm, **process_kwargs)
    X_obs = fun(da_obs, **process_kwargs)
    #< Regrid all quantiles to the RCM resolution
    logger.info(f"Regridding GCM data to RCM grid")
    X_gcm = lib.regrid(X_gcm, X_rcm)
    logger.info(f"Regridding obs data to RCM grid")
    X_obs = lib.regrid(X_obs, X_rcm)
    #< Calculate added value
    logger.info(f"Calculating added value using {distance_measure}")
    if hasattr(lib, distance_measure):
        fun = getattr(lib, distance_measure)
    else:
        assert False, f"Distance measure of {distance_measure} not implemented!"
    av = fun(X_obs, X_gcm, X_rcm)
    #< Mask data
    if not mask is None:
        mask = lib.regrid(mask, X_rcm, regrid_method="nearest_s2d")
        av = xr.where(mask, av, np.nan)
    #< Convert av to a dataset
    av = av.to_dataset(name="av")
    #< Return
    if return_X:
        return av, X_gcm, X_rcm, X_obs
    else:
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
    if args.ifiles_mask:
        logger.info(f"Opening mask")
        mask = lib.open_dataset(args.ifiles_mask)[args.varname_mask]
    else:
        mask = None

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

    #< Cut all dataarrays to same time period
    logger.info(f"Selecting time period")
    da_gcm = da_gcm.sel(time=slice(args.datestart, args.dateend))
    da_rcm = da_rcm.sel(time=slice(args.datestart, args.dateend))
    da_obs = da_obs.sel(time=slice(args.datestart, args.dateend))

    #< Select certain months
    if args.months:
        logger.info(f"Selecting months {args.months}")
        da_gcm = da_gcm.sel(time=da_gcm.time.dt.month.isin(args.months))
        da_rcm = da_rcm.sel(time=da_rcm.time.dt.month.isin(args.months))
        da_obs = da_obs.sel(time=da_obs.time.dt.month.isin(args.months))

    #< Cut all dataarrays to the same domain
    if args.lat0!=-999 and args.lat1!=-999 and args.lon0!=-999 and args.lon1!=-999:
        logger.info(f"Selecting domain")
        da_gcm = da_gcm.sel(lat=slice(args.lat0, args.lat1), lon=slice(args.lon0, args.lon1))
        da_rcm = da_rcm.sel(lat=slice(args.lat0, args.lat1), lon=slice(args.lon0, args.lon1))
        da_obs = da_obs.sel(lat=slice(args.lat0, args.lat1), lon=slice(args.lon0, args.lon1))
        if args.ifiles_mask:
            mask = mask.sel(lat=slice(args.lat0, args.lat1), lon=slice(args.lon0, args.lon1))

    #< Do masking
    if args.ifiles_mask:
        if args.op_mask == "smallerthan":
            mask = xr.where(mask <= args.value_mask, 1, 0)
        elif args.op_mask == "smaller":
            mask = xr.where(mask < args.value_mask, 1, 0)
        elif args.op_mask == "largerthan":
            mask = xr.where(mask >= args.value_mask, 1, 0)
        elif args.op_mask == "larger":
            mask = xr.where(mask > args.value_mask, 1, 0)
        elif args.op_mask == "equal":
            mask = xr.where(mask == args.value_mask, 1, 0)

    #< Calculate added value
    if args.return_X:
        av, X_gcm, X_rcm, X_obs = added_value(da_gcm, da_rcm, da_obs, args.process, args.process_kwargs, distance_measure=args.distance_measure, mask=mask, return_X=args.return_X)
    else:
        av = added_value(da_gcm, da_rcm, da_obs, args.process, args.process_kwargs, distance_measure=args.distance_measure, mask=mask, return_X=args.return_X)

    #< Save added value to netcdf
    logger.info("Saving to netcdf")
    lib.write2nc(av, args.ofile, inlogs=inlogs)
    if args.return_X:
        lib.write2nc(X_gcm, args.ofile.replace(".nc", "_X_gcm.nc"), inlogs=inlogs)
        lib.write2nc(X_rcm, args.ofile.replace(".nc", "_X_rcm.nc"), inlogs=inlogs)
        lib.write2nc(X_obs, args.ofile.replace(".nc", "_X_obs.nc"), inlogs=inlogs)


    logger.debug(f"Done")


    


if __name__ == '__main__':

    dask.config.set({
        'array.chunk-size': "512 MiB",
        'distributed.comm.timeouts.connect': '60s',
        'distributed.comm.timeouts.tcp': '60s',
        'distributed.comm.retry.count': 5,
        'distributed.scheduler.allowed-failures': 10,
    })

    parser        = parse_arguments()
    args          = parser.parse_args()
    nthreads     = args.nthreads
    nworkers     = args.nworkers

    memory_limit = '1000mb' if os.environ["HOSTNAME"].startswith("gadi-login") else int(os.environ["PBS_VMEM"]) / nworkers
    memory_limit = os.getenv('MEMORY_LIMIT', memory_limit)
    client       = dask.distributed.Client(n_workers = nworkers, threads_per_worker = nthreads,
                                           memory_limit = memory_limit, local_directory = tempfile.mkdtemp(),
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
