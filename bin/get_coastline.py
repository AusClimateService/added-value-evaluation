import numpy as np
import geopandas as gp
import argparse
from shapely.geometry import Point, MultiPolygon
from shapely.affinity import scale
from shapely.ops import cascaded_union
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from geopandas.tools import sjoin_nearest
import lib_spatial


def parse_arguments():
    # User argument input
    parser = argparse.ArgumentParser(description='Script for creating coastline shape file.')
    parser.add_argument("--region", dest='region', nargs='?', type=str, default="Australia", help="Input region using lib_spatial.py (e.g., Australia)")
    parser.add_argument("--distance", dest='distance', nargs='?', type=float, default=-100, help="Distance from coastline to use in km (default is -100). Should be negative.")
    parser.add_argument("--ofile", dest='ofile', nargs='?', type=str, default="", help="Where to save the output shapefile")
    return parser


def main():
    #< Get user arguments
    parser = parse_arguments()
    args = parser.parse_args()

    # Read the shapefile
    shapefile = lib_spatial.get_region_shape(args.region)
    print("Read shapefile")

    # Add projection
    projection = 3857 #'epsg:4326' # 3857 because it uses metres
    shapefile = shapefile.to_crs(projection)
    print("Projected shapefile")

    # Filter features for size
    filtered = []
    for polygon in shapefile.geometry:
        for geom in polygon.geoms:
            # Filter out islands with an area less then 10 km**2
            if geom.area/1e6 >= 10:
                filtered.append(geom)
    filtered = MultiPolygon(filtered)
    filtered_shapefile = gp.GeoDataFrame(geometry=[filtered])
    filtered_shapefile.crs = projection

    # Compute the coastline buffer
    buffer_distance = args.distance * 1000  # 100 km in meters
    coastline_buffer = filtered_shapefile.geometry.buffer(buffer_distance, single_sided=True)
    print("Added buffer")

    # Create the hollowed-out polygon by subtracting the point buffer
    hollowed_shapefile = shapefile.difference(coastline_buffer)
    print("Hollowed")

    # Project back to lat/lon
    hollowed_shapefile = hollowed_shapefile.to_crs(ccrs.PlateCarree())

    # hollowed_shapefile.plot()
    # plt.show()

    # Save the hollowed GeoDataFrame to a shapefile
    hollowed_shapefile.to_file(args.ofile)
    print("Saved")


if __name__ == '__main__':
    #< Call the main function
    main()