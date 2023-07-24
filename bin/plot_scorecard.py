import numpy as np
import xarray as xr
import pandas as pd
import tempfile
import argparse
import cmdline_provenance as cmdprov
import matplotlib.pyplot as plt
import time
import os
import sys
import lib
import json
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


#< Get logger
logger = lib.get_logger(__name__)


def parse_arguments():
    # User argument input
    parser = argparse.ArgumentParser(description='Script for plotting a file on a map.')
    parser.add_argument("--ifiles", dest='ifiles', nargs='+', type=str, default="", help="Input files")
    parser.add_argument("--varname", dest='varname', nargs='?', type=str, default="", help="Variable name in file")
    parser.add_argument("--names", dest='names', nargs='+', type=str, default="", help="Name for each input file (e.g. NorESM_JJA_99p)")
    parser.add_argument("--ofile", dest='ofile', nargs='?', type=str, default="", help="Path and name of output plot")
    parser.add_argument("--plt-title", dest='plt_title', nargs='?', type=str, default="", help="Title for plot")
    parser.add_argument("--plt-kwargs", dest='plt_kwargs', nargs='?', type=json.loads, default="{}", help="kwargs for plot. Instead of True/False use 1/0.")

    return parser


def main():
    # Load the logger
    logger.info(f"Start {sys.argv[0]}")

    #< Get user arguments
    parser = parse_arguments()
    args = parser.parse_args()

    #< Open datasets
    logger.info(f"Opening datasets")
    ds = []
    for f in args.ifiles:
        da = lib.open_dataset(f)[args.varname]
        ds.append( da )
    ds = xr.concat(ds, pd.Index(args.names, name="name"))
    logger.debug(ds)
    logger.debug("==========================================")

    #< Plot on scorecard
    fig = lib.heatmap(ds, xdim="quantile", ydim="name", title=args.plt_title)

    if args.ofile:
        plt.savefig(args.ofile)
    else:
        plt.show()

    #< Finish
    logger.info(f"Done")
    

if __name__ == '__main__':
    #< Set the logging level
    logger.setLevel("DEBUG")

    #< Call the main function
    main()

