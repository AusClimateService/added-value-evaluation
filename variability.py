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
    parser = argparse.ArgumentParser(description='Script for calculating variability.')
    parser.add_argument("--ifiles", dest='ifiles', nargs='*', type=str, default=[], help="Input GCM files")

    parser.add_argument("--varname", dest='varname', nargs='?', type=str, default="", help="Variable name in GCM files")

    parser.add_argument("--ifiles-mask", dest='ifiles_mask', nargs='*', type=str, default=[], help="Input masking files")
    parser.add_argument("--varname-mask", dest='varname_mask', nargs='?', type=str, default="", help="Variable name in masking files")
    parser.add_argument("--value-mask", dest='value_mask', nargs='?', type=float, default=-999, help="Value to use for masking (e.g. larger than 0")
    parser.add_argument("--op-mask", dest='op_mask', nargs='?', type=str, default="", help="Operation to use for masking (e.g. larger, smaller")

    parser.add_argument("--process", dest='process', nargs='?', type=str, default="", help="Process to get added value for (e.g., quantile)")
    parser.add_argument("--process-kwargs", dest='process_kwargs', nargs='?', type=json.loads, default="{}", help="Kwargs to pass to process function (e.g., \'{\"quantile\": 0.95}\' 0.95 for quantile)")
    parser.add_argument("--grouping", dest='grouping', nargs='?', type=str, default="", help="How to group (e.g., time.season)")
    parser.add_argument("--dim", dest='dim', nargs='?', type=str, default="", help="Dimension over which to calculate variance (e.g., time, year)")

    parser.add_argument("--datestart", dest='datestart', nargs='?', type=str, default="", help="Start date of analysis period")
    parser.add_argument("--dateend", dest='dateend', nargs='?', type=str, default="", help="End date of analysis period")
    parser.add_argument("--months", dest='months', nargs='*', type=int, default=[], help="Select only certain months (e.g. 12 1 2 for DJF")

    parser.add_argument("--ofile", dest='ofile', nargs='?', type=str, default="variability.nc", help="Path and name of output file")

    parser.add_argument("--lat0", dest='lat0', nargs='?', type=float, default=-999, help="Lower latitude to select")
    parser.add_argument("--lat1", dest='lat1', nargs='?', type=float, default=-999, help="Upper latitude to select")
    parser.add_argument("--lon0", dest='lon0', nargs='?', type=float, default=-999, help="Lower longitude to select")
    parser.add_argument("--lon1", dest='lon1', nargs='?', type=float, default=-999, help="Upper longitude to select")

    parser.add_argument("--nthreads", dest='nthreads', nargs='?', type=int, const='', default=1, help="Number of threads.")
    parser.add_argument("--nworkers", dest='nworkers', nargs='?', type=int, const='', default=2, help="Number of workers.")

    parser.add_argument("--log-level", dest='loglevel', nargs='?', type=str, default="INFO", help="Amount of log output")

    return parser



def variability(da, process, process_kwargs={}, mask=None, grouping="", dim="time"):
    """Calculate added value statistic from driving model (da_gcm), regional model (da_rcm) and reference (da_obs) dataarray

    Args:
        da (xarray dataarray): Data to calculate variability for
        process (str): Process to calculate variability for (e.g., quantile)
        process_kwargs (dict): Kwargs to pass to "process" (e.g., {'quantile':0.95})
        mask (xarray dataarray): Array (with 0 & 1) used for masking.
        grouping (str): The grouping to use (e.g., time.year)

    Returns:
        xarray dataset : Variability
    """
    #< Search for "process" function in library
    if hasattr(lib, process):
        fun = getattr(lib, process)
    else:
        assert False, f"{process} not implemented!"
    #< Calculate the "process" either for the full dataset or a grouping (i.e., a particular season)
    if grouping:
        logger.info(f"Calculating {process} for {grouping}")
        X = da.groupby(grouping).map(fun, **process_kwargs)
    else:
        X = fun(da, **process_kwargs)
    #< Calculate the variance and return
    logger.info(f"Calculating variance over {dim}")
    X = X.var(dim)
    return X


def main():
    # Load the logger
    logger.info(f"Start")

    #< Get user arguments
    parser = parse_arguments()
    args = parser.parse_args()

    #< Open datasets
    logger.info(f"Opening dataset")
    ds = lib.open_dataset(args.ifiles)
    logger.debug(ds)
    if args.ifiles_mask:
        logger.info(f"Opening mask")
        mask = lib.open_dataset(args.ifiles_mask)[args.varname_mask]
    else:
        mask = None

    #< Get the history of the input files
    inlogs = {}
    if "history" in ds.attrs:
        inlogs[f"input"] = ds.attrs["history"]

    #< Fetch variable from dataset
    logger.info(f"Extracting variable from dataset")
    da = ds[args.varname]

    #< Cut all dataarray to the time period
    logger.info(f"Selecting time period")
    da = da.sel(time=slice(args.datestart, args.dateend))

    #< Select certain months
    if args.months:
        logger.info(f"Selecting months {args.months}")
        da = da.sel(time=da.time.dt.month.isin(args.months))

    #< Cut all dataarrays to the same domain
    if args.lat0!=-999 and args.lat1!=-999 and args.lon0!=-999 and args.lon1!=-999:
        logger.info(f"Selecting domain")
        da = da.sel(lat=slice(args.lat0, args.lat1), lon=slice(args.lon0, args.lon1))
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

    #< Calculate variability
    var = variability(da, args.process, args.process_kwargs, mask=mask, grouping=args.grouping, dim=args.dim)

    #< Save added value to netcdf
    logger.info("Saving to netcdf")
    lib.write2nc(var, args.ofile, inlogs=inlogs)

    #< Finish
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
