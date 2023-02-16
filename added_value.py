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

    parser.add_argument("--quantile", dest='quantile', nargs='?', type=float, default="", help="Quantile to calculate added value for")

    parser.add_argument("--datestart", dest='datestart', nargs='?', type=str, default="", help="Start date of analysis period")
    parser.add_argument("--dateend", dest='dateend', nargs='?', type=str, default="", help="End date of analysis period")

    parser.add_argument("--ofile", dest='ofile', nargs='?', type=str, default="av.nc", help="Path and name of output file")

    parser.add_argument("--lat0", dest='lat0', nargs='?', type=float, default=-999, help="Lower latitude to select")
    parser.add_argument("--lat1", dest='lat1', nargs='?', type=float, default=-999, help="Upper latitude to select")
    parser.add_argument("--lon0", dest='lon0', nargs='?', type=float, default=-999, help="Lower longitude to select")
    parser.add_argument("--lon1", dest='lon1', nargs='?', type=float, default=-999, help="Upper longitude to select")

    parser.add_argument("--nthreads", dest='nthreads', nargs='?', type=int, const='', default=1, help="Number of threads.")
    parser.add_argument("--nworkers", dest='nworkers', nargs='?', type=int, const='', default=2, help="Number of workers.")

    parser.add_argument("--log-level", dest='loglevel', nargs='?', type=str, default="INFO", help="Amount of log output")

    return parser



def added_value(da_gcm, da_rcm, da_obs, quantile, measure="AVrmse", mask=None):
    """Calculate added value statistic from driving model (da_gcm), regional model (da_rcm) and reference (da_obs) dataarray

    Args:
        da_gcm (xarray dataarray): Driving model data
        da_rcm (xarray dataarray): Regional model data
        da_obs (xarray dataarray): Reference data
        measure (str): Distance measure to use for added value calculation
        quantile (float): Quantile to calculate added value for (e.g., 0.9)
        mask (xarray dataarray): Array (with 0 & 1) used for masking.

    Returns:
        xarray dataset : Added value
    """

    #< Re-chunk the data because quantiles cannot be calculated over chunked dimensions
    logger.info(f"Re-chunking data")
    da_gcm = da_gcm.chunk({"time":-1, "lat":"auto"})
    da_rcm = da_rcm.chunk({"time":-1, "lat":"auto"})
    da_obs = da_obs.chunk({"time":-1, "lat":"auto"})
    #< Calculate quantile
    logger.info(f"Calculating {quantile*100}th quantile")
    X_gcm = da_gcm.quantile(quantile,"time", skipna=True).load()
    X_rcm = da_rcm.quantile(quantile,"time", skipna=True).load()
    X_obs = da_obs.quantile(quantile,"time", skipna=True).load()
    #< Regrid all quantiles to the RCM resolution
    logger.info(f"Regridding GCM data to RCM grid")
    X_gcm = lib.regrid(X_gcm, X_rcm)
    logger.info(f"Regridding obs data to RCM grid")
    X_obs = lib.regrid(X_obs, X_rcm)
    #< Calculate added value
    logger.info(f"Calculating added value using {measure}")
    if hasattr(lib, measure):
        fun = getattr(lib, measure)
    else:
        assert False, f"Distance measure of {measure} not implemented!"
    av = fun(X_obs, X_gcm, X_rcm)
    #< Mask data
    if mask:
        av = xr.where(mask, av, np.nan)
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

    #< Cut all dataarrays to same time period
    logger.info(f"Selecting time period")
    da_gcm = da_gcm.sel(time=slice(args.datestart, args.dateend))
    da_rcm = da_rcm.sel(time=slice(args.datestart, args.dateend))
    da_obs = da_obs.sel(time=slice(args.datestart, args.dateend))

    #< Cut all dataarrays to the same domain
    logger.info(f"Selecting domain")
    da_gcm = da_gcm.sel(lat=slice(args.lat0, args.lat1), lon=slice(args.lon0, args.lon1))
    da_rcm = da_rcm.sel(lat=slice(args.lat0, args.lat1), lon=slice(args.lon0, args.lon1))
    da_obs = da_obs.sel(lat=slice(args.lat0, args.lat1), lon=slice(args.lon0, args.lon1))

    #< Calculate added value
    av = added_value(da_gcm, da_rcm, da_obs, args.quantile, measure="AVrmse")

    #< Save added value to netcdf
    logger.info("Saving to netcdf")
    lib.write2nc(av, args.ofile, inlogs=inlogs)

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