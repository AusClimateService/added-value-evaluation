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
import logging

#< Get logger
logger = lib.get_logger(__name__)


def parse_arguments():
    # User argument input
    parser = argparse.ArgumentParser(description='Script for calculating normalised added value using AV and obs as input.')
    parser.add_argument("--ifile-av", dest='ifile_av', nargs='*', type=str, default=[], help="Input added value file")
    parser.add_argument("--ifile-var", dest='ifile_var', nargs='*', type=str, default=[], help="Input variability file")

    parser.add_argument("--varname-av", dest='varname_av', nargs='?', type=str, default="av", help="Variable name in added value file")
    parser.add_argument("--varname-var", dest='varname_var', nargs='?', type=str, default="", help="Variable name in variability files")

    parser.add_argument("--ofile", dest='ofile', nargs='?', type=str, default="av.nc", help="Path and name of output file")

    parser.add_argument("--nthreads", dest='nthreads', nargs='?', type=int, const='', default=1, help="Number of threads.")
    parser.add_argument("--nworkers", dest='nworkers', nargs='?', type=int, const='', default=2, help="Number of workers.")

    parser.add_argument("--log-level", dest='loglevel', nargs='?', type=str, default="INFO", help="Amount of log output")

    return parser



def added_value_norm(da_av, da_var):
    """Calculate normalised added value statistic from added value (da_av) and reference variability (da_var) dataarray

    Args:
        da_av (xarray dataarray): Added value data
        da_var (xarray dataarray): Reference variability data

    Returns:
        xarray dataset : Normalised added value
    """
    #< Regrid all varibility to the RCM resolution
    logger.info(f"Regridding variance data to RCM grid")
    da_var = lib.regrid(da_var, da_av)
    #< Calculate normalised added value
    logger.info(f"Calculating normalised added value")
    eps = 1e-12
    da_var = da_var.where(da_var>eps, np.nan)
    av_norm = da_av / np.sqrt(da_var)
    #< Convert av to a dataset
    av_norm = av_norm.to_dataset(name="av_norm")
    #< Return
    return av_norm


def main():
    # Load the logger
    logger.info(f"Start {sys.argv[0]}")

    #< Get user arguments
    parser = parse_arguments()
    args = parser.parse_args()

    #< Open datasets
    logger.info(f"Opening datasets")
    ds_av = lib.open_dataset(args.ifile_av)
    ds_var = lib.open_dataset(args.ifile_var)
    logger.debug(ds_av)
    logger.debug(ds_var)

    #< Get the history of the input files
    inlogs = {}
    if "history" in ds_av.attrs:
        inlogs[f"av"] = ds_av.attrs["history"]
    if "history" in ds_var.attrs:
        inlogs[f"var"] = ds_var.attrs["history"]

    #< Fetch variable from dataset
    logger.info(f"Extracting variables from datasets")
    da_av = ds_av[args.varname_av]
    da_var = ds_var[args.varname_var]

    #< Calculate realised added value
    av_norm = added_value_norm(da_av, da_var)

    #< Save realised added value to netcdf
    logger.info("Saving to netcdf")
    lib.write2nc(av_norm, args.ofile, inlogs=inlogs)

    #< Finish
    logger.info(f"Done")


    


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
