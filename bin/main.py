import glob
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
import cmip6_interface
import era5_interface
import barpa_drs_interface
import agcd_interface
import ccam_drs_interface
from subprocess import Popen, PIPE


#< Get logger
logger = lib.get_logger(__name__)


def parse_arguments():
    # User argument input
    parser = argparse.ArgumentParser(description='Script for calculating added value using GCM, RCM and observations as input.')
    parser.add_argument("--gcm", dest='gcm', nargs='?', type=str, default="", help="Input GCM")
    parser.add_argument("--rcm", dest='rcm', nargs='?', type=str, default="", help="Input RCM")
    parser.add_argument("--obs", dest='obs', nargs='?', type=str, default="", help="Input reference")
    parser.add_argument("--scenario-hist", dest='scenario_hist', nargs='?', type=str, default="", help="Input scenario historical")
    parser.add_argument("--scenario-fut", dest='scenario_fut', nargs='?', type=str, default="", help="Input scenario future")
    parser.add_argument("--freq", dest='freq', nargs='?', type=str, default="day", help="Input frequency")
    parser.add_argument("--variable", dest='variable', nargs='?', type=str, default="", help="Variable")
    parser.add_argument("--regions", dest='regions', nargs='*', type=str, default=[], help="Regions to select")
    parser.add_argument("--seasons", dest='seasons', nargs='*', type=str, default=[], help="Seasons to select")
    parser.add_argument("--odir", dest='odir', nargs='?', type=str, default=".", help="Output directory")
    parser.add_argument("--overwrite", dest='overwrite',  action='store_true', help="Overwrite existing files.")

    parser.add_argument("--av-measures", dest='av_measures', nargs='+', type=str, default="", choices=["added_value", "potential_added_value", "realised_added_value", "variability", "added_value_norm"], help="Calculate added_value, potential_added_value, realised_added_value or variability")
    parser.add_argument("--process", dest='process', nargs='?', type=str, default="", help="Process to get added value for (e.g., quantile)")
    parser.add_argument("--process-kwargs", dest='process_kwargs', nargs='?', type=str, default="{}", help="Kwargs to pass to process function (e.g., \'{\"quantile\": 0.95}\' 0.95 for quantile)")
    parser.add_argument("--av-distance-measure", dest='av_distance_measure', nargs='?', type=str, default="", help="Distance measure to use for AV calculation")
    parser.add_argument("--pav-distance-measure", dest='pav_distance_measure', nargs='?', type=str, default="", help="Distance measure to use for PAV calculation")

    parser.add_argument("--datestart-hist", dest='datestart_hist', nargs='?', type=str, default="", help="Start date of historical analysis period")
    parser.add_argument("--dateend-hist", dest='dateend_hist', nargs='?', type=str, default="", help="End date of historical analysis period")
    parser.add_argument("--datestart-fut", dest='datestart_fut', nargs='?', type=str, default="", help="Start date of future analysis period")
    parser.add_argument("--dateend-fut", dest='dateend_fut', nargs='?', type=str, default="", help="End date of future analysis period")
    parser.add_argument("--months", dest='months', nargs='*', type=int, default=[], help="Select only certain months (e.g. 12 1 2 for DJF")

    parser.add_argument("--lat0", dest='lat0', nargs='?', type=float, default=-999, help="Lower latitude to select")
    parser.add_argument("--lat1", dest='lat1', nargs='?', type=float, default=-999, help="Upper latitude to select")
    parser.add_argument("--lon0", dest='lon0', nargs='?', type=float, default=-999, help="Lower longitude to select")
    parser.add_argument("--lon1", dest='lon1', nargs='?', type=float, default=-999, help="Upper longitude to select")

    parser.add_argument("--return-X", dest='return_X', action='store_true', help="Also return the regridded climate statistics")
    parser.add_argument("--agcd_mask", dest='agcd_mask', action='store_true', help='Whether to apply masking for AGCD precipitation data')

    parser.add_argument("--nthreads", dest='nthreads', nargs='?', type=int, const='', default=1, help="Number of threads.")
    parser.add_argument("--nworkers", dest='nworkers', nargs='?', type=int, const='', default=2, help="Number of workers.")
    parser.add_argument("--nprocs", dest='nprocs', nargs='?', type=int, const='', default=1, help="Number of parallel processes.")

    parser.add_argument("--log-level", dest='loglevel', nargs='?', type=str, default="INFO", help="Amount of log output")

    return parser


def driving_model_loader(gcm, scen, freq, var):
    if gcm in ["ACCESS-ESM1-5", "ACCESS-CM2", "EC-Earth3", "NorESM2-MM", "CMCC-ESM2", "CESM2"]:
        return cmip6_interface.get_cmip6_files(gcm, scen, freq, var), var
    elif gcm in ["ERA5"]:
        era5_varname_dict = {
            "tasmax": "tas",
            "tasmin": "tas",
            "pr": "pr",
        }
        era5_filenmame_dict = {
            "tasmax": "tasmax",
            "tasmin": "tasmin",
            "pr": "prmean",
        }
        file_list = glob.glob(f"/g/data/tp28/ERA5/{era5_filenmame_dict[var]}_era5_oper_sfc_*.nc")
        return file_list, era5_varname_dict[var]
    else:
        raise Exception(f"Driving model {gcm} for {scen} not found!")


def rcm_model_loader(gcm, rcm, scen, freq, var):
    if rcm == "BARPA-R":
        barpa_name_dict = {
            "ERA5": "ECMWF-ERA5",
            "ACCESS-ESM1-5": "CSIRO-ACCESS-ESM1-5",
            "ACCESS-CM2": "CSIRO-ARCCSS-ACCESS-CM2",
            "EC-Earth3": "EC-Earth-Consortium-EC-Earth3",
            "NorESM2-MM": "NCC-NorESM2-MM",
            "CMCC-ESM2": "CMCC-CMCC-ESM2",
            "CESM2": "NCAR-CESM2",
        }
        if gcm == "ERA5":
            scen = "evaluation"
        return barpa_drs_interface.get_barpa_files(barpa_name_dict[gcm], scen, freq, var), var
    elif rcm == "CCAM":
        ccam_name_dict = {
            "ERA5": "ECMWF-ERA5",
            "ACCESS-CM2": "CSIRO-ARCCSS-ACCESS-CM2",
            "ACCESS-ESM1-5": "CSIRO-ACCESS-ESM1-5",
            "EC-Earth3": "EC-Earth-Consortium-EC-Earth3",
            "CESM2": "NCAR-CESM2",
            "CMCC-ESM2": "CMCC-CMCC-ESM2",
            "CNRM-ESM2-1": "CNRM-CERFACS-CNRM-ESM2-1",
            "NorESM2-MM": "NCC-NorESM2-MM",
        }
        return ccam_drs_interface.get_ccam_files(ccam_name_dict[gcm], scen, freq, var), var
    else:
        logger.error(f"RCM {rcm} is not implemented!")
        exit(1)


def ref_loader(obs, freq, var):
    if obs == "AGCD":
        agcd_freq_dict = {
            "day": "daily",
        }
        agcd_varname_dict = {
            "tasmax": "tmax",
            "tasmin": "tmin",
            "pr": "precip",
        }
        subversion='total' if var == 'pr' else "mean"
        return agcd_interface.get_files(agcd_freq_dict[freq], agcd_varname_dict[var], subversion=subversion), agcd_varname_dict[var]
    else:
        logger.error(f"Observations {args.obs} is not implemented!")
        exit(1)


def cmd_split(cmd, split=" "):
    cmd_out = cmd.split(split)
    del_index = []
    for i in range(len(cmd_out)):
        #< Remove empty entries
        if cmd_out[i] == "":
            del_index.append(i)
        #< Remove newline entries
        if cmd_out[i] == "\n":
            del_index.append(i)
    #< Remove from the back otherwise indexing issues
    for index in sorted(del_index, reverse=True):
        del cmd_out[index]
    #< Remove any \n in the cmd list
    for i in range(len(cmd_out)):
        if "\n" in cmd_out[i]:
            cmd_out[i] = cmd_out[i].replace("\n","")
    return cmd_out


def season_loader(season=""):
    season_pairs = ""
    if season:
        season_dict = {
            "annual": "",
            "DJF": {
                "--months": "12 1 2"
            },
            "MAM": {
                "--months": "3 4 5"
            },
            "JJA": {
                "--months": "6 7 8"
            },
            "SON": {
                "--months": "9 10 11"
            },
        }
        season_pairs = [(key, season_dict[season][key]) for key in season_dict[season]]
        season_pairs = " ".join("(%s,%s)" % tup for tup in season_pairs).replace("(","").replace(")","").replace(","," ")
    return season_pairs


def loop_av(args, gcm_files, rcm_files, obs_files, gcm_varname, rcm_varname, obs_varname, ofile, return_X=False, agcd_mask=False):

    #< Get the season
    season_cmd = season_loader(season=args.season)

    #< Get the cmd command
    cmd = f"""
    python added_value.py --ifiles-gcm {' '.join(gcm_files)} --ifiles-rcm {' '.join(rcm_files)} --ifiles-obs {' '.join(obs_files)} 
    --varname-gcm {gcm_varname}  --varname-rcm {rcm_varname} --varname-obs {obs_varname} 
    --region {args.region} 
    {season_cmd} 
    --process {args.process} --process-kwargs {args.process_kwargs} --distance-measure {args.av_distance_measure}
    --datestart {args.datestart_hist} --dateend {args.dateend_hist}
    --nworkers {args.nworkers} --nthreads {args.nthreads}
    --lat0 {args.lat0} --lat1 {args.lat1}
    --lon0 {args.lon0} --lon1 {args.lon1}
    --ofile {ofile}
    --log-level {args.loglevel}
    """

    if agcd_mask:
        cmd += " --agcd_mask"
    if return_X:
        cmd += " --return_X"
        
    cmd = cmd_split(cmd)
    logger.debug(cmd)
    return Popen(cmd)


def loop_av_norm(args, ofile_av, ofile_var, av_varname, var_varname, ofile, return_X=False, agcd_mask=False):
    #< Get the cmd command
    cmd = f"""
    python added_value_norm.py --ifile-av {ofile_av} --ifile-var {ofile_var} 
    --varname-av {av_varname} --varname-var {var_varname} 
    --nworkers {args.nworkers} --nthreads {args.nthreads}
    --ofile {ofile}
    --log-level {args.loglevel}
    --return-X False
    """

    if obs_varname == "precip":
        cmd = cmd + """
    --ifile-obs-premask /g/data/tp28/cst565/ACS_added_value/agcd_mask_0p8_larger1_1mask_0nomask.nc
    --varname-obs-premask mask
    """
    cmd = cmd_split(cmd)
    logger.debug(cmd)
    return Popen(cmd)


def loop_av_norm(args, gcm_hist_files, gcm_fut_files, rcm_hist_files, rcm_fut_files, gcm_varname, rcm_varname, ofile, return_X=False, agcd_mask=False):
    #< Get the cmd command
    cmd = f"""
    python added_value_norm.py --ifile-av {ofile_av} --ifile-var {ofile_var} 
    --varname-av {av_varname} --varname-var {var_varname} 
    --nworkers {args.nworkers} --nthreads {args.nthreads}
    --ofile {ofile}
    --log-level {args.loglevel}
    """

    if agcd_mask:
        cmd += " --agcd_mask"
    if return_X:
        cmd += " --return_X"

    cmd = cmd_split(cmd)
    logger.debug(cmd)
    return Popen(cmd)



def loop_pav(args, gcm_hist_files, gcm_fut_files, rcm_hist_files, rcm_fut_files, gcm_varname, rcm_varname, ofile, return_X=False, agcd_mask=False):
    #< Get the season
    season_cmd = season_loader(season=args.season)

    #< Get the cmd command
    cmd = f"""
    python potential_added_value.py --ifiles-gcm-hist {' '.join(gcm_hist_files)} --ifiles-rcm-hist {' '.join(rcm_hist_files)} 
    --ifiles-gcm-fut {' '.join(gcm_fut_files)} --ifiles-rcm-fut {' '.join(rcm_fut_files)} 
    --varname-gcm {gcm_varname}  --varname-rcm {rcm_varname} 
    --region {args.region} 
    {season_cmd} 
    --process {args.process} --process-kwargs {args.process_kwargs} --distance-measure {args.pav_distance_measure}
    --datestart-hist {args.datestart_hist} --dateend-hist {args.dateend_hist}
    --datestart-fut {args.datestart_fut} --dateend-fut {args.dateend_fut}
    --nworkers {args.nworkers} --nthreads {args.nthreads}
    --lat0 {args.lat0} --lat1 {args.lat1}
    --lon0 {args.lon0} --lon1 {args.lon1}
    --ofile {ofile}
    --log-level {args.loglevel}
    """

    if agcd_mask:
        cmd += " --agcd_mask"
    if return_X:
        cmd += " --return_X"

    cmd = cmd_split(cmd)
    logger.debug(cmd)
    return Popen(cmd)


def loop_var(args, obs_files, obs_varname, ofile, grouping="", dim=""):

    #< Get the season
    season_cmd = season_loader(season=args.season)

    #< Get the cmd command
    cmd = f"""
    python variability.py --ifiles {' '.join(obs_files)} 
    --varname {obs_varname} 
    --region {args.region} 
    {season_cmd} 
    --process {args.process} --process-kwargs {args.process_kwargs}
    --grouping {grouping} --dim {dim}
    --datestart {args.datestart_hist} --dateend {args.dateend_hist} 
    --nworkers {args.nworkers} --nthreads {args.nthreads} 
    --lat0 {args.lat0} --lat1 {args.lat1} 
    --lon0 {args.lon0} --lon1 {args.lon1} 
    --ofile {ofile} 
    --log-level {args.loglevel} 
    """
    if obs_varname == "precip":
        cmd = cmd + """
    --ifile-mask /g/data/tp28/cst565/ACS_added_value/agcd_mask_0p8_larger1_1mask_0nomask.nc
    --varname-mask mask
    """
    cmd = cmd_split(cmd)
    logger.debug(cmd)
    return Popen(cmd)


def loop_rav(args, ofile_av, ofile_pav, ofile_var, av_varname, pav_varname, var_varname, ofile):
    #< Get the cmd command
    cmd = f"""
    python realised_added_value.py --ifile-av {ofile_av} --ifile-pav {ofile_pav} --ifile-var {ofile_var} 
    --varname-av {av_varname}  --varname-pav {pav_varname} --varname-var {var_varname} 
    --nworkers {args.nworkers} --nthreads {args.nthreads}
    --ofile {ofile}
    --log-level {args.loglevel}
    """
    cmd = cmd_split(cmd)
    logger.debug(cmd)
    return Popen(cmd)


def check_processes(processes, args, trigger=False):
    if len(processes) >= args.nprocs or trigger:
        logger.debug(f"Triggering {len(processes)} processes")
        for i, p in enumerate(processes[::-1]):
            p.wait()
            output, error = p.communicate()
            if p.returncode!=0:
                logger.error(f"check_process returned {p.returncode} exit code!")
                exit(p.returncode)
            processes.pop(-1)
            


def tidy_filename(filename):
    #< Replace double underscore with single underscore
    while "__" in filename:
        filename = filename.replace("__", "_")
    return filename

def get_ofile(odir="./", measure="", variable="", gcm="", scenario="", rcm="", obs="", freq="", region="", season="", datestart_hist="", dateend_hist="", datestart_fut="", dateend_fut="", kwargs=dict()):
    # Create a list of kwargs in the style of [key0, val0, key1, val1, ... keyN, valN]
    kwargs_list = []
    for key in kwargs:
        kwargs_list.append(key)
        kwargs_list.append(kwargs[key])
    ofile = f"{odir}/{measure}_{variable}_{'_'.join([str(i).replace('.','p') for i in kwargs_list])}_{gcm}_{scenario}_{rcm}_{obs}_{freq}_{region.replace(' ', '_')}_{season}"
    if datestart_hist and dateend_hist:
        ofile = ofile + f"_{datestart_hist}-{dateend_hist}"
    if datestart_fut and dateend_fut:
        ofile = ofile + f"_{datestart_fut}-{dateend_fut}"
    ofile = ofile + ".nc"
    ofile = tidy_filename(ofile)
    return ofile


def main():
    # Load the logger
    logger.info(f"Start {sys.argv[0]}")

    #< Get user arguments
    parser = parse_arguments()
    args = parser.parse_args()

    #< Get the input files
    #< GCM files
    gcm_files_hist, gcm_varname = driving_model_loader(args.gcm, args.scenario_hist, args.freq, args.variable)
    gcm_files_fut, gcm_varname = driving_model_loader(args.gcm, args.scenario_fut, args.freq, args.variable)
    rcm_files_hist, rcm_varname = rcm_model_loader(args.gcm, args.rcm, args.scenario_hist, args.freq, args.variable)
    rcm_files_fut, rcm_varname = rcm_model_loader(args.gcm, args.rcm, args.scenario_fut, args.freq, args.variable)
    obs_files, obs_varname = ref_loader(args.obs, args.freq, args.variable)
    logger.debug(f"Found these driving model files:")
    logger.debug(gcm_files_hist)
    logger.debug("=================================")
    logger.debug(gcm_files_fut)
    logger.debug(f"Found these rcm files:")
    logger.debug(rcm_files_hist)
    logger.debug("=================================")
    logger.debug(rcm_files_fut)
    logger.debug(f"Found these reference files:")
    logger.debug(obs_files)


    if not os.path.exists(args.odir):
        os.makedirs(args.odir, exist_ok=True)
    processes = []
    ### Added value calculation
    #< Construct added value output filename
    for season in args.seasons:
        args.season = season
        for region in args.regions:
            args.region = region
            for av_measure in args.av_measures:
                ofile_av = get_ofile(odir=args.odir, measure=args.av_distance_measure, variable=args.variable, gcm=args.gcm, scenario=args.scenario_hist, rcm=args.rcm, obs=args.obs, freq=args.freq, region=args.region, season=args.season, datestart_hist=args.datestart_hist, dateend_hist=args.dateend_hist, kwargs=json.loads(args.process_kwargs))
                #< Check if we need to calculate AV
                if (av_measure == "added_value" or av_measure == "realised_added_value" or av_measure == "added_value_norm") and (not os.path.isfile(ofile_av) or args.overwrite):
                    #< Collect processes
                    processes.append( loop_av(args, gcm_files_hist, rcm_files_hist, obs_files, gcm_varname, rcm_varname, obs_varname, ofile_av, return_X=args.return_X, agcd_mask=args.agcd_mask) )
                #< Check if processes are to be triggered
                check_processes(processes, args)

    ### Potential added value calculation
    #< Construct potential added value output filename
    for season in args.seasons:
        args.season = season
        for region in args.regions:
            args.region = region
            for av_measure in args.av_measures:
                ofile_pav = get_ofile(odir=args.odir, measure=args.pav_distance_measure, variable=args.variable, gcm=args.gcm, scenario=args.scenario_fut, rcm=args.rcm, freq=args.freq, region=args.region, season=args.season, datestart_hist=args.datestart_hist, dateend_hist=args.dateend_hist, datestart_fut=args.datestart_fut, dateend_fut=args.dateend_fut, kwargs=json.loads(args.process_kwargs))
                #< Check if we need to calculate PAV
                if (av_measure == "potential_added_value" or av_measure == "realised_added_value") and (not os.path.isfile(ofile_pav) or args.overwrite):
                    #< Collect processes
                    processes.append( loop_pav(args, gcm_files_hist, gcm_files_fut, rcm_files_hist, rcm_files_fut, gcm_varname, rcm_varname, ofile_pav, return_X=args.return_X, agcd_mask=args.agcd_mask) )
                #< Check if processes are to be triggered
                check_processes(processes, args)

    ### Variability calculation
    #< Construct variability output filename
    for season in args.seasons:
        args.season = season
        for region in args.regions:
            args.region = region
            for av_measure in args.av_measures:
                grouping = ""
                variability_dim = ""
                if args.process == "quantile":
                    grouping = "time.year"
                    variability_dim = "year"
                ofile_var = get_ofile(odir=args.odir, measure="VAR", variable=args.variable, obs=args.obs, freq=args.freq, region=args.region, season=args.season, datestart_hist=args.datestart_hist, dateend_hist=args.dateend_hist, kwargs=json.loads(args.process_kwargs))
                #< Check if we need to calculate VAR
                if av_measure == "variability" or av_measure == "realised_added_value" or av_measure == "added_value_norm" and (not os.path.isfile(ofile_var) or args.overwrite):
                    #< Collect processes
                    processes.append( loop_var(args, obs_files, obs_varname, ofile_var, grouping=grouping, dim=variability_dim) )
                #< Check if processes are to be triggered
                check_processes(processes, args)
    #< Trigger any left over processes before we start with realised added value
    check_processes(processes, args, True)


    ### Normalised added value calculation
    for season in args.seasons:
        args.season = season
        for region in args.regions:
            args.region = region
            for av_measure in args.av_measures:
                ofile_av = get_ofile(odir=args.odir, measure=args.av_distance_measure, variable=args.variable, gcm=args.gcm, scenario=args.scenario_hist, rcm=args.rcm, obs=args.obs, freq=args.freq, region=args.region, season=args.season, datestart_hist=args.datestart_hist, dateend_hist=args.dateend_hist, kwargs=json.loads(args.process_kwargs))
                ofile_var = get_ofile(odir=args.odir, measure="VAR", variable=args.variable, obs=args.obs, freq=args.freq, region=args.region, season=args.season, datestart_hist=args.datestart_hist, dateend_hist=args.dateend_hist, kwargs=json.loads(args.process_kwargs))
                ofile_av_norm = get_ofile(odir=args.odir, measure=f"{args.av_distance_measure}_norm", variable=args.variable, gcm=args.gcm, scenario=args.scenario_hist, rcm=args.rcm, obs=args.obs, freq=args.freq, region=args.region, season=args.season, datestart_hist=args.datestart_hist, dateend_hist=args.dateend_hist, kwargs=json.loads(args.process_kwargs))
                #< Check if we need to calculate VAR
                if av_measure == "added_value_norm" and not os.path.isfile(ofile_av_norm):
                    #< Collect processes
                    processes.append( loop_av_norm(args, ofile_av, ofile_var, av_varname="av", var_varname=obs_varname, ofile=ofile_av_norm) )
                #< Check if processes are to be triggered
                check_processes(processes, args)
    #< Trigger any left over processes
    check_processes(processes, args, True)


    ### Realised added calculation
    #< Construct variability output filename
    for season in args.seasons:
        args.season = season
        for region in args.regions:
            args.region = region
            for av_measure in args.av_measures:
                ofile_av = get_ofile(odir=args.odir, measure=args.av_distance_measure, variable=args.variable, gcm=args.gcm, scenario=args.scenario_hist, rcm=args.rcm, obs=args.obs, freq=args.freq, region=args.region, season=args.season, datestart_hist=args.datestart_hist, dateend_hist=args.dateend_hist, kwargs=json.loads(args.process_kwargs))
                ofile_pav = get_ofile(odir=args.odir, measure=args.pav_distance_measure, variable=args.variable, gcm=args.gcm, scenario=args.scenario_fut, rcm=args.rcm, freq=args.freq, region=args.region, season=args.season, datestart_hist=args.datestart_hist, dateend_hist=args.dateend_hist, datestart_fut=args.datestart_fut, dateend_fut=args.dateend_fut, kwargs=json.loads(args.process_kwargs))
                ofile_var = get_ofile(odir=args.odir, measure="VAR", variable=args.variable, obs=args.obs, freq=args.freq, region=args.region, season=args.season, datestart_hist=args.datestart_hist, dateend_hist=args.dateend_hist, kwargs=json.loads(args.process_kwargs))
                ofile_rav = get_ofile(odir=args.odir, measure="RAV", variable=args.variable, gcm=args.gcm, scenario=args.scenario_fut, rcm=args.rcm, obs=args.obs, freq=args.freq, region=args.region, season=args.season, datestart_hist=args.datestart_hist, dateend_hist=args.dateend_hist, datestart_fut=args.datestart_fut, dateend_fut=args.dateend_fut, kwargs=json.loads(args.process_kwargs))
                #< Check if we need to calculate VAR
                if av_measure == "realised_added_value" and not os.path.isfile(ofile_rav):
                    #< Collect processes
                    processes.append( loop_rav(args, ofile_av, ofile_pav, ofile_var, av_varname="av", pav_varname="pav", var_varname=obs_varname, ofile=ofile_rav) )
                #< Check if processes are to be triggered
                check_processes(processes, args)
    #< Trigger any left over processes
    check_processes(processes, args, True)


if __name__ == '__main__':
    #< Get user arguments
    parser = parse_arguments()
    args = parser.parse_args()

    #< Set the logging level
    logger.setLevel(args.loglevel)

    #< Call the main function
    main()

