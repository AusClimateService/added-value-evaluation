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
    parser.add_argument("--ifile", dest='ifile', nargs='?', type=str, default="", help="Input file")
    parser.add_argument("--varname", dest='varname', nargs='?', type=str, default="", help="Variable name in file")
    parser.add_argument("--ofile", dest='ofile', nargs='?', type=str, default="av.nc", help="Path and name of output file")
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
    da = lib.open_dataset(args.ifile)[args.varname]
    logger.debug(da)
    logger.debug("==========================================")

    #< Plot on map
    p = da.plot.pcolormesh(transform=ccrs.PlateCarree(), subplot_kws=dict(projection=ccrs.PlateCarree(central_longitude=180)), **args.plt_kwargs)
    ax = p.axes
    # Map settings
    ax.set_title(args.plt_title)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.coastlines()
    plt.show()

    #< Finish
    logger.info(f"Done")


    


if __name__ == '__main__':
    #< Set the logging level
    logger.setLevel("DEBUG")

    #< Call the main function
    main()

