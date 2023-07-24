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
    parser = argparse.ArgumentParser(description='Script for calculating potential added value using GCM and RCM as input.')
    parser.add_argument("--ifiles-gcm-hist", dest='ifiles_gcm_hist', nargs='*', type=str, default=[], help="Input historical GCM files")
    parser.add_argument("--ifiles-rcm-hist", dest='ifiles_rcm_hist', nargs='*', type=str, default=[], help="Input historical RCM files")
    parser.add_argument("--ifiles-gcm-fut", dest='ifiles_gcm_fut', nargs='*', type=str, default=[], help="Input scenario GCM files")
    parser.add_argument("--ifiles-rcm-fut", dest='ifiles_rcm_fut', nargs='*', type=str, default=[], help="Input scenario RCM files")

    parser.add_argument("--varname-gcm", dest='varname_gcm', nargs='?', type=str, default="", help="Variable name in GCM files")
    parser.add_argument("--varname-rcm", dest='varname_rcm', nargs='?', type=str, default="", help="Variable name in RCM files")

    parser.add_argument("--region", dest='region', nargs='?', type=str, default="", help="Region masking using lib_spatial.py")

    parser.add_argument("--agcd_mask", dest='agcd_mask', action='store_true', help='Whether to apply masking for AGCD precipitation data')

    parser.add_argument("--process", dest='process', nargs='?', type=str, default="", help="Process to get added value for (e.g., quantile)")
    parser.add_argument("--process-kwargs", dest='process_kwargs', nargs='?', type=json.loads, default="{}", help="Kwargs to pass to process function (e.g., \'{\"quantile\": 0.95}\' 0.95 for quantile)")
    parser.add_argument("--distance-measure", dest='distance_measure', nargs='?', type=str, default="", help="Distance measure to use for PAV calculation, either PAVdiff or PAVdiff_rel")

    parser.add_argument("--datestart-hist", dest='datestart_hist', nargs='?', type=str, default="", help="Start date of historical period")
    parser.add_argument("--dateend-hist", dest='dateend_hist', nargs='?', type=str, default="", help="End date of historical period")
    parser.add_argument("--datestart-fut", dest='datestart_fut', nargs='?', type=str, default="", help="Start date of future ssp period")
    parser.add_argument("--dateend-fut", dest='dateend_fut', nargs='?', type=str, default="", help="End date of future ssp period")

    parser.add_argument("--months", dest='months', nargs='*', type=int, default=[], help="Select only certain months (e.g. 12 1 2 for DJF")

    parser.add_argument("--ofile", dest='ofile', nargs='?', type=str, default="pav.nc", help="Path and name of output file")

    parser.add_argument("--lat0", dest='lat0', nargs='?', type=float, default=-999, help="Lower latitude to select")
    parser.add_argument("--lat1", dest='lat1', nargs='?', type=float, default=-999, help="Upper latitude to select")
    parser.add_argument("--lon0", dest='lon0', nargs='?', type=float, default=-999, help="Lower longitude to select")
    parser.add_argument("--lon1", dest='lon1', nargs='?', type=float, default=-999, help="Upper longitude to select")

    parser.add_argument("--return-X", dest='return_X', action='store_true', help="Also return the regridded climate statistics")

    parser.add_argument("--nthreads", dest='nthreads', nargs='?', type=int, const='', default=1, help="Number of threads.")
    parser.add_argument("--nworkers", dest='nworkers', nargs='?', type=int, const='', default=2, help="Number of workers.")

    parser.add_argument("--log-level", dest='loglevel', nargs='?', type=str, default="INFO", help="Amount of log output")

    return parser



def potential_added_value(da_gcm_hist, da_gcm_fut, da_rcm_hist, da_rcm_fut, process, process_kwargs={}, distance_measure="PAVdiff", region=None, return_X=False):
    """Calculate potential added value statistic from driving model (da_gcm) and regional model (da_rcm) dataarray

    Args:
        da_gcm_hist (xarray dataarray): Driving model data for historical period
        da_gcm_fut (xarray dataarray): Driving model data for ssp scenario
        da_rcm_hist (xarray dataarray): Regional model data for historical period
        da_rcm_fut (xarray dataarray): Regional model data for ssp scenario
        process (str): Process to calculate PAV for (e.g., quantile)
        process_kwargs (dict): Kwargs to pass to "process" (e.g., {'quantile':0.95})
        distance_measure (str): Distance measure to use for added value calculation. 
                                The function needs to be defined in lib.py.
        region (str): Use lib_spatial for masking.
        write_X (bool): Should the regridded climate statistic be written out too?
        agcd_mask (bool): Apply AGCD mask based on AGCD precip data

    Returns:
        xarray dataset : Added value
    """
    #< Make sure all dataarrays have the same units
    assert "units" in da_gcm_hist.attrs, f"da_gcm_hist has no units attribute"
    assert "units" in da_rcm_hist.attrs, f"da_rcm_hist has no units attribute"
    assert "units" in da_gcm_fut.attrs, f"da_gcm_fut has no units attribute"
    assert "units" in da_rcm_fut.attrs, f"da_rcm_fut has no units attribute"
    assert da_gcm_hist.attrs["units"] == da_gcm_fut.attrs["units"] == da_rcm_hist.attrs["units"] == da_rcm_fut.attrs["units"], f"Not all dataarrays have the same units: {da_gcm_hist.attrs['units']} != {da_gcm_fut.attrs['units']} != {da_rcm_hist.attrs['units']} != {da_rcm_fut.attrs['units']}"
    #< Search for "process" function in library and run it on all three dataarrays
    if hasattr(lib, process):
        fun = getattr(lib, process)
    else:
        assert False, f"{process} not implemented!"
    logger.info(f"Calculating {process}")
    X_gcm_hist = fun(da_gcm_hist, **process_kwargs)
    X_gcm_fut = fun(da_gcm_fut, **process_kwargs)
    X_rcm_hist = fun(da_rcm_hist, **process_kwargs)
    X_rcm_fut = fun(da_rcm_fut, **process_kwargs)
    # Get the response
    X_gcm = X_gcm_fut - X_gcm_hist
    X_rcm = X_rcm_fut - X_rcm_hist
    #< Regrid all quantiles to the RCM resolution
    logger.info(f"Regridding GCM data to RCM grid")
    X_gcm = lib.regrid(X_gcm, X_rcm)
    #< Mask data
    if not region is None:
        logger.info("Masking X_gcm.")
        X_gcm = lib_spatial.apply_region_mask(X_gcm, region)
        logger.info("Masking X_rcm.")
        X_rcm = lib_spatial.apply_region_mask(X_rcm, region)
    #< AGCD mask
    if agcd_mask:
        logger.info("Masking X_gcm with AGCD mask")
        X_gcm = lib_spatial.apply_agcd_data_mask(X_gcm)
        logger.info("Masking X_rcm with AGCD mask")
        X_rcm = lib_spatial.apply_agcd_data_mask(X_rcm)
    #< Calculate added value
    logger.info(f"Calculating potential added value using {distance_measure}")
    if hasattr(lib, distance_measure):
        fun = getattr(lib, distance_measure)
    else:
        assert False, f"Distance measure of {distance_measure} not implemented!"
    pav = fun(X_gcm, X_rcm)
    #< Mask data
    if not region is None:
        pav = lib_spatial.apply_region_mask(pav, region.replace("_", " "))
    #< Convert av to a dataset
    pav = pav.to_dataset(name="pav")
    #< Return
    if return_X:
        return pav, X_gcm, X_rcm
    else:
        return pav


def main():
    # Load the logger
    logger.info(f"Start {sys.argv[0]}")

    #< Get user arguments
    parser = parse_arguments()
    args = parser.parse_args()

    #< Open datasets
    logger.info(f"Opening datasets")
    ds_gcm_hist = lib.open_dataset(args.ifiles_gcm_hist)
    ds_rcm_hist = lib.open_dataset(args.ifiles_rcm_hist)
    ds_gcm_fut = lib.open_dataset(args.ifiles_gcm_fut)
    ds_rcm_fut = lib.open_dataset(args.ifiles_rcm_fut)
    logger.debug(ds_gcm_hist)
    logger.debug(ds_rcm_hist)
    logger.debug(ds_gcm_fut)
    logger.debug(ds_rcm_fut)

    #< Get the history of the input files
    inlogs = {}
    if "history" in ds_gcm_hist.attrs:
        inlogs[f"gcm_hist"] = ds_gcm_hist.attrs["history"]
    if "history" in ds_rcm_hist.attrs:
        inlogs[f"rcm_hist"] = ds_rcm_hist.attrs["history"]
    if "history" in ds_gcm_fut.attrs:
        inlogs[f"gcm_fut"] = ds_gcm_fut.attrs["history"]
    if "history" in ds_rcm_fut.attrs:
        inlogs[f"rcm_fut"] = ds_rcm_fut.attrs["history"]

    #< Fetch variable from dataset
    logger.info(f"Extracting variables from datasets")
    da_gcm_hist = ds_gcm_hist[args.varname_gcm]
    da_gcm_fut = ds_gcm_fut[args.varname_gcm]
    da_rcm_hist = ds_rcm_hist[args.varname_rcm]
    da_rcm_fut = ds_rcm_fut[args.varname_rcm]

    #< Cut all dataarrays to same time period
    logger.info(f"Selecting ssp time period")
    da_gcm_hist = da_gcm_hist.sel(time=slice(args.datestart_hist, args.dateend_hist))
    da_gcm_fut = da_gcm_fut.sel(time=slice(args.datestart_fut, args.dateend_fut))
    da_rcm_hist = da_rcm_hist.sel(time=slice(args.datestart_hist, args.dateend_hist))
    da_rcm_fut = da_rcm_fut.sel(time=slice(args.datestart_fut, args.dateend_fut))

    #< Select certain months
    if args.months:
        logger.info(f"Selecting months {args.months}")
        da_gcm_hist = da_gcm_hist.sel(time=da_gcm_hist.time.dt.month.isin(args.months))
        da_gcm_fut = da_gcm_fut.sel(time=da_gcm_fut.time.dt.month.isin(args.months))
        da_rcm_hist = da_rcm_hist.sel(time=da_rcm_hist.time.dt.month.isin(args.months))
        da_rcm_fut = da_rcm_fut.sel(time=da_rcm_fut.time.dt.month.isin(args.months))

    #< Cut all dataarrays to the same domain
    if args.lat0!=-999 and args.lat1!=-999 and args.lon0!=-999 and args.lon1!=-999:
        logger.info(f"Selecting domain")
        da_gcm_hist = da_gcm_hist.sel(lat=slice(args.lat0, args.lat1), lon=slice(args.lon0, args.lon1))
        da_gcm_fut = da_gcm_fut.sel(lat=slice(args.lat0, args.lat1), lon=slice(args.lon0, args.lon1))
        da_rcm_hist = da_rcm_hist.sel(lat=slice(args.lat0, args.lat1), lon=slice(args.lon0, args.lon1))
        da_rcm_fut = da_rcm_fut.sel(lat=slice(args.lat0, args.lat1), lon=slice(args.lon0, args.lon1))

    #< Calculate added value
    if args.return_X:
        pav, X_gcm, X_rcm = potential_added_value(da_gcm_hist, da_gcm_fut, da_rcm_hist, da_rcm_fut, args.process, args.process_kwargs, distance_measure=args.distance_measure, region=args.region, return_X=args.return_X)
    else:
        pav = potential_added_value(da_gcm_hist, da_gcm_fut, da_rcm_hist, da_rcm_fut, args.process, args.process_kwargs, distance_measure=args.distance_measure, region=args.region, return_X=args.return_X)

    #< Save added value to netcdf
    logger.info("Saving to netcdf")
    lib.write2nc(pav, args.ofile, inlogs=inlogs)
    if args.return_X:
        lib.write2nc(X_gcm, args.ofile.replace(".nc", "_X_gcm.nc"), inlogs=inlogs)
        lib.write2nc(X_rcm, args.ofile.replace(".nc", "_X_rcm.nc"), inlogs=inlogs)


    logger.info(f"Done")


    


if __name__ == '__main__':

    dask.config.set({
        'array.chunk-size': "512 MiB",
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
