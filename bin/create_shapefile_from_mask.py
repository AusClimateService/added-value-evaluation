import numpy as np
import geopandas as gp
import argparse
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import lib
from shapely.geometry import Polygon, MultiPolygon
from skimage import measure


def parse_arguments():
    # User argument input
    parser = argparse.ArgumentParser(description='Script for creating coastline shape file.')
    parser.add_argument("--mask-file", dest='mask_file', nargs='?', type=str, default="", help="Input masking file (True/False) to use for creating a shapefile.")
    parser.add_argument("--mask-value", dest='mask_value', nargs='?', type=float, default=500, help="Value used to create the mask. Set values large than this to True; False otherwise.")
    parser.add_argument("--filter-area-size", dest='filter_area', nargs='?', type=float, default=0, help="Filter areas that are smaller than this (in km**2).")
    parser.add_argument("--overlay", dest='overlay', nargs='?', type=str, default="", help="Overlay (intersect) another shapefile with this one.")
    parser.add_argument("--ofile", dest='ofile', nargs='?', type=str, default="", help="Where to save the output shapefile")
    return parser


def get_lat_lon(da, ilat, ilon):
    ilat = int(np.round(ilat))
    ilon = int(np.round(ilon))
    return (float(da["lon"][ilon].values), float(da["lat"][ilat].values))


def main():
    #< Get user arguments
    parser = parse_arguments()
    args = parser.parse_args()

    # Read masking file
    mask = lib.open_dataset(args.mask_file)["orog"].load()
    print("Read masking file")
    print(mask)

    # Set values larger than 500 to true
    mask = xr.where(mask>args.mask_value, True, False)
    mask.plot.pcolormesh()

    # Find contours at a constant value of 0.99 (1 does not work)
    # This returns the index values of the contour line
    contours = measure.find_contours(mask.values, 0.99)

    # Convert to list of polygons
    polygons = []
    for contour in contours:
        coords = [get_lat_lon(mask, c[0], c[1]) for c in contour]
        polygons.append(Polygon(coords))
    polygons = MultiPolygon(polygons)

    # Create shapefile
    gdf = gp.GeoDataFrame(geometry=[polygons])
    gdf.crs = ccrs.PlateCarree()
    gdf = gdf.to_crs('3857')

    # Filter features for size
    filtered = []
    for polygon in gdf.geometry:
        for geom in polygon.geoms:
            # Filter out smaller than args.filter_by_area km**2
            if geom.area/1e6 >= args.filter_area:
                filtered.append(geom)
    filtered = MultiPolygon(filtered)
    filtered_shapefile = gp.GeoDataFrame(geometry=[filtered])
    filtered_shapefile.crs = "3857"
    filtered_shapefile = filtered_shapefile.to_crs('epsg:4326')

    # Overlay
    if args.overlay:
        shapefile2 = gp.read_file(args.overlay)
        filtered_shapefile = filtered_shapefile.to_crs(shapefile2.crs)
        filtered_shapefile = gp.overlay(filtered_shapefile, shapefile2, how='intersection')

        fig, ax = plt.subplots()
        shapefile2.plot(ax=ax, color='blue', label="overlay")
        filtered_shapefile.plot(ax=ax, color='red', label="intersection")
        plt.show()

    # Save the GeoDataFrame as a shapefile
    filtered_shapefile.to_file(args.ofile)


if __name__ == '__main__':
    #< Call the main function
    main()