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
    parser = argparse.ArgumentParser(description='Script for calculating variability.')
    parser.add_argument("--ifiles", dest='ifiles', nargs='*', type=str, default=[], help="Input reference files")
    parser.add_argument("--varname", dest='varname', nargs='?', type=str, default="", help="Variable name in reference files")

    parser.add_argument("--region", dest='region', nargs='?', type=str, default="", help="Region masking using lib_spatial.py")
    parser.add_argument("--agcd_mask", dest='agcd_mask', action='store_true', help='Whether to apply masking for AGCD precipitation data')

    parser.add_argument("--process", dest='process', nargs='?', type=str, default="", help="Process to get added value for (e.g., quantile)")
    parser.add_argument("--process-kwargs", dest='process_kwargs', nargs='?', type=json.loads, default="{}", help="Kwargs to pass to process function (e.g., \'{\"quantile\": 0.95}\' 0.95 for quantile)")
    parser.add_argument("--grouping", dest='grouping', nargs='?', type=str, default="", const="", help="How to group (e.g., time.season)")
    parser.add_argument("--dim", dest='dim', nargs='?', type=str, default="", help="Dimension over which to calculate variance (e.g., time, year)")

    parser.add_argument("--datestart", dest='datestart', nargs='?', type=str, default="", help="Start date of analysis period")
    parser.add_argument("--dateend", dest='dateend', nargs='?', type=str, default="", help="End date of analysis period")
    parser.add_argument("--months", dest='months', nargs='*', type=int, default=[], help="Select only certain months (e.g. 12 1 2 for DJF")

    parser.add_argument("--ofile", dest='ofile', nargs='?', type=str, default="variability.nc", help="Path and name of output file")

    parser.add_argument("--upscale2gcm", default=False, action='store_true',help="Upscale the RCM and to the GCM resolution instead.")
    parser.add_argument("--ifile-gcm", nargs='?', type=str, default=None, help="GCM file to use for upscaling")
    parser.add_argument("--upscale2ref", default=False, action='store_true',help="Upscale the RCM and OBS to reference resolution instead.")
    parser.add_argument("--ifile-ref-grid", nargs='?', type=str, default=None, help="Reference file to use for upscaling")


    parser.add_argument("--lat0", dest='lat0', nargs='?', type=float, default=-999, help="Lower latitude to select")
    parser.add_argument("--lat1", dest='lat1', nargs='?', type=float, default=-999, help="Upper latitude to select")
    parser.add_argument("--lon0", dest='lon0', nargs='?', type=float, default=-999, help="Lower longitude to select")
    parser.add_argument("--lon1", dest='lon1', nargs='?', type=float, default=-999, help="Upper longitude to select")

    parser.add_argument("--nthreads", dest='nthreads', nargs='?', type=int, const='', default=1, help="Number of threads.")
    parser.add_argument("--nworkers", dest='nworkers', nargs='?', type=int, const='', default=2, help="Number of workers.")

    parser.add_argument("--log-level", dest='loglevel', nargs='?', type=str, default="INFO", help="Amount of log output")

    return parser



def variability(da, process, process_kwargs={}, grouping="", dim="time", agcd_mask=False, region=None, upscale2gcm=False, da_gcm=None, upscale2ref=False, da_ref_grid=None):
    """Calculate added value statistic from driving model (da_gcm), regional model (da_rcm) and reference (da_obs) dataarray

    Args:
        da (xarray dataarray): Data to calculate variability for
        process (str): Process to calculate variability for (e.g., quantile)
        process_kwargs (dict): Kwargs to pass to "process" (e.g., {'quantile':0.95})
        agcd_mask (bool): Apply AGCD mask based on AGCD precip data
        grouping (str): The grouping to use (e.g., time.year)
        upscale2gcm (bool): Upscale to GCM resolution
        da_gcm (xarray): Reference GCM data to use for upscaling
        upscale2ref (bool): Should all variables be upscaled to a reference grid resolution?
        da_ref_grid (xarray): Reference grid to use for upscaling?

    Returns:
        xarray dataset : Variability
    """
    assert not (upscale2gcm and upscale2ref), f"upscale2gcm and upscale2ref cannot both be True!"
    #< Mask data
    if not region is None:
        logger.info(f"Masking {region}.")
        da = lib_spatial.apply_region_mask(da, region.replace("_", " "))
        logger.debug(da)
    #< AGCD mask
    if agcd_mask:
        logger.info("Masking with AGCD mask")
        da = lib_spatial.apply_agcd_data_mask(da)

    #< Maybe upscale
    if upscale2gcm:
        da = da.chunk({"time": "auto", "lat": -1, "lon": -1})
        logger.info(f"Upscaling to GCM grid")
        da = lib.regrid(da, da_gcm, reuse_regrid_weights=True)
        logger.debug(da)
    elif upscale2ref:
        da = da.chunk({"time": "auto", "lat": -1, "lon": -1})
        logger.info(f"Upscaling to reference grid")
        da = lib.regrid(da, da_ref_grid, reuse_regrid_weights=True)

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
    logger.info(f"Start {sys.argv[0]}")

    #< Get user arguments
    parser = parse_arguments()
    args = parser.parse_args()

    assert not (args.upscale2gcm and args.upscale2ref), f"upscale2gcm and upscale2ref cannot both be True!"

    #< Open datasets
    logger.info(f"Opening dataset")
    ds = lib.open_dataset(args.ifiles)
    logger.debug(ds)
    logger.debug("Input dataset looks like:")
    logger.debug(ds)
    da_gcm = None
    if args.upscale2gcm:
        da_gcm = lib.open_dataset(args.ifile_gcm)
        logger.debug("GCM reference file looks like:")
        logger.debug(da_gcm)
    da_ref_grid = None
    if args.upscale2ref:
        da_ref_grid = xr.open_dataarray(args.ifile_ref_grid)
        logger.debug("Reference grid looks like:")
        logger.debug(da_ref_grid)
    logger.debug("----------------------------------------------")

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
    logger.debug(da)
    logger.debug("----------------------------------------------")

    #< Select certain months
    if args.months:
        logger.info(f"Selecting months {args.months}")
        da = da.sel(time=da.time.dt.month.isin(args.months))
        logger.debug(da)
        logger.debug("----------------------------------------------")

    #< Cut all dataarrays to the same domain
    if args.lat0!=-999 and args.lat1!=-999 and args.lon0!=-999 and args.lon1!=-999:
        logger.info(f"Selecting domain")
        da = da.sel(lat=slice(args.lat0, args.lat1), lon=slice(args.lon0, args.lon1))
        if args.upscale2ref:
                da_ref_grid = da_ref_grid.sel(lat=slice(args.lat0, args.lat1), lon=slice(args.lon0, args.lon1))
                logger.debug(da_ref_grid)

    #< Calculate variability
    var = variability(
            da,
            args.process,
            args.process_kwargs,
            grouping=args.grouping,
            dim=args.dim,
            agcd_mask=args.agcd_mask,
            upscale2gcm=args.upscale2gcm,
            da_gcm = da_gcm,
            upscale2ref=args.upscale2ref,
            da_ref_grid=da_ref_grid
    )

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
        "distributed.scheduler.worker-saturation": 1.1, #< This should use the new behaviour which helps with memory pile up
        'array.slicing.split_large_chunks': False,
    })

    parser        = parse_arguments()
    args          = parser.parse_args()
    nthreads     = args.nthreads
    nworkers     = args.nworkers

    memory_limit = '1000mb' if os.environ["HOSTNAME"].startswith("gadi-login") else int(os.environ["PBS_VMEM"]) / nworkers
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
