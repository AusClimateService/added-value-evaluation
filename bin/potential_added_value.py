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

    parser.add_argument("--reuse-X", dest='reuse_X', action='store_true', help="Reuse the regridded climate statistics")
    parser.add_argument("--ifile-X-gcm-hist", nargs='?', type=str, default=None, help="Input statistic historical GCM file")
    parser.add_argument("--ifile-X-gcm-fut", nargs='?', type=str, default=None, help="Input statistic future GCM file")
    parser.add_argument("--ifile-X-rcm-hist", nargs='?', type=str, default=None, help="Input statistic historical RCM file")
    parser.add_argument("--ifile-X-rcm-fut", nargs='?', type=str, default=None, help="Input statistic future RCM file")

    parser.add_argument("--upscale2gcm", default=False, action='store_true',help="Upscale the RCM and to the GCM resolution instead.")
    parser.add_argument("--upscale2ref", default=False, action='store_true',help="Upscale the RCM and OBS to reference resolution instead.")
    parser.add_argument("--ifile-ref-grid", nargs='?', type=str, default=None, help="Reference file to use for upscaling")

    parser.add_argument("--nthreads", dest='nthreads', nargs='?', type=int, const='', default=1, help="Number of threads.")
    parser.add_argument("--nworkers", dest='nworkers', nargs='?', type=int, const='', default=2, help="Number of workers.")

    parser.add_argument("--log-level", dest='loglevel', nargs='?', type=str, default="INFO", help="Amount of log output")

    return parser



def potential_added_value(da_gcm_hist, da_gcm_fut, da_rcm_hist, da_rcm_fut, process, process_kwargs={}, distance_measure="PAVdiff", region=None, reuse_X=False, agcd_mask=False,
                        ifile_X_rcm_hist=None, ifile_X_rcm_fut=None, ifile_X_gcm_hist=None, ifile_X_gcm_fut=None, upscale2gcm=False, upscale2ref=False, da_ref_grid=None):
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
        upscale2ref (bool): Should all variables be upscaled to a reference grid resolution?
        da_ref_grid (xarray): Reference grid to use for upscaling?

    Returns:
        xarray dataset : Added value
    """
    assert not (upscale2gcm and upscale2ref), f"upscale2gcm and upscale2ref cannot both be True!"
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

    
    if reuse_X and os.path.isfile(ifile_X_rcm_hist):
        logger.info(f"Using existing {ifile_X_rcm_hist} for RCM ")
        X_rcm_hist = xr.open_dataarray(ifile_X_rcm_hist)
        logger.debug(X_rcm_hist)
    elif not reuse_X or not os.path.isfile(ifile_X_rcm_hist):
        if upscale2gcm:
            #< AGCD mask
            if agcd_mask:
                logger.info("Masking historical RCM with AGCD mask")
                da_rcm_hist = lib_spatial.apply_agcd_data_mask(da_rcm_hist)
                logger.debug(da_rcm_hist)
            da_rcm_hist = da_rcm_hist.chunk({"time": "auto", "lat": -1, "lon": -1})
            logger.info(f"Upscaling historical RCM to GCM grid")
            da_rcm_hist_up = lib.regrid(da_rcm_hist, da_gcm_hist, reuse_regrid_weights=True)
            #< Mask data
            if not region is None:
                logger.info("Masking historical RCM.")
                da_rcm_hist_up = lib_spatial.apply_region_mask(da_rcm_hist_up, region.replace("_", " "))
                logger.debug(da_rcm_hist_up)
            logger.info(f"Calculating {process} for upscaled historical RCM")
            X_rcm_hist = fun(da_rcm_hist_up, **process_kwargs)
        elif upscale2ref:
            #< AGCD mask
            if agcd_mask:
                logger.info("Masking historical RCM with AGCD mask")
                da_rcm_hist = lib_spatial.apply_agcd_data_mask(da_rcm_hist)
                logger.debug(da_rcm_hist)
            da_rcm_hist = da_rcm_hist.chunk({"time": "auto", "lat": -1, "lon": -1})
            logger.info(f"Upscaling historical RCM to reference grid")
            da_rcm_hist_up = lib.regrid(da_rcm_hist, da_ref_grid, reuse_regrid_weights=True)
            #< Mask data
            if not region is None:
                logger.info("Masking historical RCM.")
                da_rcm_hist_up = lib_spatial.apply_region_mask(da_rcm_hist_up, region.replace("_", " "))
                logger.debug(da_rcm_hist_up)
            logger.info(f"Calculating {process} for upscaled historical RCM")
            X_rcm_hist = fun(da_rcm_hist_up, **process_kwargs)
        else:
            #< Mask data
            if not region is None:
                logger.info("Masking")
                da_rcm_hist = lib_spatial.apply_region_mask(da_rcm_hist, region.replace("_", " "))
                logger.debug(da_rcm_hist)
            #< AGCD mask
            if agcd_mask:
                logger.info("Masking da_rcm_hist with AGCD mask")
                da_rcm_hist = lib_spatial.apply_agcd_data_mask(da_rcm_hist)
                logger.debug(da_rcm_hist)
            logger.info(f"Calculating {process} for RCM")
            X_rcm_hist = fun(da_rcm_hist, **process_kwargs)
    if reuse_X and not os.path.isfile(ifile_X_rcm_hist):
        logger.info(f"Saving {ifile_X_rcm_hist} for RCM to netcdf")
        lib.write2nc(X_rcm_hist, ifile_X_rcm_hist)


    if reuse_X and os.path.isfile(ifile_X_rcm_fut):
        logger.info(f"Using existing {ifile_X_rcm_fut} for RCM ")
        X_rcm_fut = xr.open_dataarray(ifile_X_rcm_fut)
        logger.debug(X_rcm_fut)
    elif not reuse_X or not os.path.isfile(ifile_X_rcm_fut):
        if upscale2gcm:
            #< AGCD mask
            if agcd_mask:
                logger.info("Masking future RCM with AGCD mask")
                da_rcm_fut = lib_spatial.apply_agcd_data_mask(da_rcm_fut)
                logger.debug(da_rcm_fut)
            da_rcm_fut = da_rcm_fut.chunk({"time": "auto", "lat": -1, "lon": -1})
            logger.info(f"Upscaling future RCM to GCM grid")
            da_rcm_fut_up = lib.regrid(da_rcm_fut, da_gcm_fut, reuse_regrid_weights=True)
            #< Mask data
            if not region is None:
                logger.info("Masking future RCM.")
                da_rcm_fut_up = lib_spatial.apply_region_mask(da_rcm_fut_up, region.replace("_", " "))
                logger.debug(da_rcm_fut_up)
            logger.info(f"Calculating {process} for upscaled future RCM")
            X_rcm_fut = fun(da_rcm_fut_up, **process_kwargs)
        elif upscale2ref:
            #< AGCD mask
            if agcd_mask:
                logger.info("Masking future RCM with AGCD mask")
                da_rcm_fut = lib_spatial.apply_agcd_data_mask(da_rcm_fut)
                logger.debug(da_rcm_fut)
            da_rcm_fut = da_rcm_fut.chunk({"time": "auto", "lat": -1, "lon": -1})
            logger.info(f"Upscaling future RCM to reference grid")
            da_rcm_fut_up = lib.regrid(da_rcm_fut, da_ref_grid, reuse_regrid_weights=True)
            #< Mask data
            if not region is None:
                logger.info("Masking future RCM.")
                da_rcm_fut_up = lib_spatial.apply_region_mask(da_rcm_fut_up, region.replace("_", " "))
                logger.debug(da_rcm_fut_up)
            logger.info(f"Calculating {process} for upscaled future RCM")
            X_rcm_fut = fun(da_rcm_fut_up, **process_kwargs)
        else:
            #< Mask data
            if not region is None:
                logger.info("Masking da_rcm_fut.")
                da_rcm_fut = lib_spatial.apply_region_mask(da_rcm_fut, region.replace("_", " "))
                logger.debug(da_rcm_fut)
            #< AGCD mask
            if agcd_mask:
                logger.info("Masking da_rcm_fut with AGCD mask")
                da_rcm_fut = lib_spatial.apply_agcd_data_mask(da_rcm_fut)
                logger.debug(da_rcm_fut)
            logger.info(f"Calculating {process} for RCM")
            X_rcm_fut = fun(da_rcm_fut, **process_kwargs)
    if reuse_X and not os.path.isfile(ifile_X_rcm_fut):
        logger.info(f"Saving {ifile_X_rcm_fut} for RCM to netcdf")
        lib.write2nc(X_rcm_fut, ifile_X_rcm_fut)
    

    if reuse_X and os.path.isfile(ifile_X_gcm_hist):
        logger.info(f"Using existing {ifile_X_gcm_hist} for GCM ")
        X_gcm_hist = xr.open_dataarray(ifile_X_gcm_hist)
        logger.debug(X_gcm_hist)
    elif not reuse_X or not os.path.isfile(ifile_X_gcm_hist):
        if upscale2gcm:
            #< Mask data
            if not region is None:
                logger.info("Masking historical GCM.")
                da_gcm_hist_masked = lib_spatial.apply_region_mask(da_gcm_hist, region.replace("_", " "))
                logger.debug(da_gcm_hist_masked)
            #< AGCD mask
            if agcd_mask:
                logger.info("Masking historical GCM with AGCD mask")
                da_gcm_hist_masked = lib_spatial.apply_agcd_data_mask(da_gcm_hist_masked)
                logger.debug(da_gcm_hist_masked)
            logger.info(f"Calculating {process} for historical GCM")
            X_gcm_hist = fun(da_gcm_hist_masked, **process_kwargs)
        elif upscale2ref:
            #< AGCD mask
            if agcd_mask:
                logger.info("Masking future GCM with AGCD mask")
                da_gcm_hist = lib_spatial.apply_agcd_data_mask(da_gcm_hist)
                logger.debug(da_gcm_hist)
            da_gcm_hist = da_gcm_hist.chunk({"time": "auto", "lat": -1, "lon": -1})
            logger.info(f"Upscaling future GCM to reference grid")
            da_gcm_hist_up = lib.regrid(da_gcm_hist, da_ref_grid, reuse_regrid_weights=True)
            #< Mask data
            if not region is None:
                logger.info("Masking historical GCM.")
                da_gcm_hist_up = lib_spatial.apply_region_mask(da_gcm_hist_up, region.replace("_", " "))
                logger.debug(da_gcm_hist_up)
            logger.info(f"Calculating {process} for upscaled future GCM")
            X_gcm_hist = fun(da_gcm_hist_up, **process_kwargs)
        else:
            #< AGCD mask
            if agcd_mask:
                logger.info("Masking da_gcm_hist_up with AGCD mask")
                da_gcm_hist = lib_spatial.apply_agcd_data_mask(da_gcm_hist)
                logger.debug(da_gcm_hist)
            logger.info(f"Regridding GCM data to RCM grid")
            da_gcm_hist_up = lib.regrid(da_gcm_hist, da_rcm_hist)
            #< Mask data
            if not region is None:
                logger.info("Masking da_gcm_hist_up.")
                da_gcm_hist_up = lib_spatial.apply_region_mask(da_gcm_hist_up, region.replace("_", " "))
                logger.debug(da_gcm_hist_up)
            logger.info(f"Calculating {process} for GCM")
            X_gcm_hist = fun(da_gcm_hist_up, **process_kwargs)
            
    if reuse_X and not os.path.isfile(ifile_X_gcm_hist):
        logger.info(f"Saving {ifile_X_gcm_hist} for GCM to netcdf")
        lib.write2nc(X_gcm_hist, ifile_X_gcm_hist)
    

    if reuse_X and os.path.isfile(ifile_X_gcm_fut):
        logger.info(f"Using existing {ifile_X_gcm_fut} for GCM ")
        X_gcm_fut = xr.open_dataarray(ifile_X_gcm_fut)
        logger.debug(X_gcm_fut)
    elif not reuse_X or not os.path.isfile(ifile_X_gcm_fut):
        if upscale2gcm:
            #< Mask data
            if not region is None:
                logger.info("Masking future GCM.")
                da_gcm_fut_masked = lib_spatial.apply_region_mask(da_gcm_fut, region.replace("_", " "))
                logger.debug(da_gcm_fut_masked)
            #< AGCD mask
            if agcd_mask:
                logger.info("Masking future GCM with AGCD mask")
                da_gcm_fut_masked = lib_spatial.apply_agcd_data_mask(da_gcm_fut_masked)
                logger.debug(da_gcm_fut_masked)
            logger.info(f"Calculating {process} for future GCM")
            X_gcm_fut = fun(da_gcm_fut_masked, **process_kwargs)
        elif upscale2ref:
            #< AGCD mask
            if agcd_mask:
                logger.info("Masking future GCM with AGCD mask")
                da_gcm_fut = lib_spatial.apply_agcd_data_mask(da_gcm_fut)
                logger.debug(da_gcm_fut)
            da_gcm_fut = da_gcm_fut.chunk({"time": "auto", "lat": -1, "lon": -1})
            logger.info(f"Upscaling future GCM to reference grid")
            da_gcm_fut_up = lib.regrid(da_gcm_fut, da_ref_grid, reuse_regrid_weights=True)
            #< Mask data
            if not region is None:
                logger.info("Masking future GCM.")
                da_gcm_fut_up = lib_spatial.apply_region_mask(da_gcm_fut_up, region.replace("_", " "))
                logger.debug(da_gcm_fut_up)
            logger.info(f"Calculating {process} for upscaled future GCM")
            X_gcm_fut = fun(da_gcm_fut_up, **process_kwargs)
        else:
            logger.info(f"Regridding GCM data to RCM grid")
            da_gcm_fut_up = lib.regrid(da_gcm_fut, da_rcm_fut)
            #< Mask data
            if not region is None:
                logger.info("Masking da_gcm_fut_up.")
                da_gcm_fut_up = lib_spatial.apply_region_mask(da_gcm_fut_up, region.replace("_", " "))
                logger.debug(da_gcm_fut_up)
            #< AGCD mask
            if agcd_mask:
                logger.info("Masking da_gcm_fut_up with AGCD mask")
                da_gcm_fut_up = lib_spatial.apply_agcd_data_mask(da_gcm_fut_up)
                logger.debug(da_gcm_fut_up)
            logger.info(f"Calculating {process} for GCM")
            X_gcm_fut = fun(da_gcm_fut_up, **process_kwargs)
    if reuse_X and not os.path.isfile(ifile_X_gcm_fut):
        logger.info(f"Saving {ifile_X_gcm_fut} for GCM to netcdf")
        lib.write2nc(X_gcm_fut, ifile_X_gcm_fut)

    # Calculate change
    X_gcm = X_gcm_fut - X_gcm_hist
    X_rcm = X_rcm_fut - X_rcm_hist
    

    #< Calculate potential added value
    logger.info(f"Calculating potential added value using {distance_measure}")
    if hasattr(lib, distance_measure):
        fun = getattr(lib, distance_measure)
    else:
        assert False, f"Distance measure of {distance_measure} not implemented!"
    pav = fun(X_gcm, X_rcm)
    #< Convert av to a dataset
    pav = pav.to_dataset(name="pav")
    #< Return
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

    da_ref_grid = None
    if args.upscale2ref:
        da_ref_grid = xr.open_dataarray(args.ifile_ref_grid)
        logger.debug(da_ref_grid)

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
        if args.upscale2ref:
                da_ref_grid = da_ref_grid.sel(lat=slice(args.lat0, args.lat1), lon=slice(args.lon0, args.lon1))
                logger.debug(da_ref_grid)

    #< Calculate added value
    pav = potential_added_value(
        da_gcm_hist,
        da_gcm_fut,
        da_rcm_hist,
        da_rcm_fut,
        args.process,
        args.process_kwargs,
        distance_measure=args.distance_measure,
        region=args.region,
        reuse_X=args.reuse_X,
        agcd_mask=args.agcd_mask,
        ifile_X_rcm_hist=args.ifile_X_rcm_hist,
        ifile_X_rcm_fut=args.ifile_X_rcm_fut,
        ifile_X_gcm_hist=args.ifile_X_gcm_hist,
        ifile_X_gcm_fut=args.ifile_X_gcm_fut,
        upscale2gcm=args.upscale2gcm,
        upscale2ref=args.upscale2ref,
        da_ref_grid=da_ref_grid,
    )

    #< Save added value to netcdf
    logger.info("Saving to netcdf")
    lib.write2nc(pav, args.ofile, inlogs=inlogs)

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
