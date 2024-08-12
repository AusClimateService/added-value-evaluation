import numpy as np
import xarray as xr

# values taken from /g/data/lp01/CMIP6/CMIP/NCC/NorESM2-MM/historical/r1i1p1f1/Amon/pr/gr1.5/v20191108
# and cut to CORDEX-Australasia domain.
lat0 = -44.25
lat1 = -9.75
lon0 = 112.5
lon1 = 154.5
dx = 1.5

lats = np.round(np.arange(lat0, lat1+dx, dx),2)
lons = np.round(np.arange(lon0,lon1+dx,dx),1)
data = np.ones((len(lats), len(lons)))

da_ref_grid = xr.DataArray(
    data=data,
    dims=["lat", "lon"],
    coords=dict(
        lon=(["lon"], lons),
        lat=(["lat"], lats)
    )
)
ds_ref_grid = da_ref_grid.to_dataset(name="data")

da_ref_grid.to_netcdf(f"ref_grid_{dx}deg.nc")