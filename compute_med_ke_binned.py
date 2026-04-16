#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 10:50:48 2025

@author: yan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
useful for eke_mean_std_over_cycles.py

"""

import argparse
from pathlib import Path
import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import shapely
from shapely.geometry import box
from shapely import contains_xy
from shapely.validation import make_valid


def build_med_mask():
    med_bbox   = box(-6, 30, 36, 46)
    black_sea  = box(27, 41, 42, 47)     
    bay_biscay = box(-6, 43, -1, 46)     
    roi = shapely.difference(med_bbox, shapely.unary_union([black_sea, bay_biscay]))
    roi = make_valid(roi)          
    return roi.buffer(0)           
'''
# ---------- ROI : Mer d'Alboran ----------
def build_med_mask():
    """
    ROI = Mer d'Alboran uniquement (deg, -180..180)
    Ajuste les bornes si tu veux élargir/rétrécir un peu.
    """
    alboran_bbox = box(-6.0, 34.0, -0.5, 38.0)
    roi = make_valid(alboran_bbox)
    return roi.buffer(0)
'''
ROI_POLY = build_med_mask()

def to_m180_180(lon):
    return ((lon + 180) % 360) - 180

def make_grid(lon_min=-6, lon_max=36, lat_min=30, lat_max=46, res=0.1):
    lon_edges = np.arange(lon_min, lon_max + 1e-9, res)
    lat_edges = np.arange(lat_min, lat_max + 1e-9, res)
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    return lon_edges, lat_edges, lon_centers, lat_centers

def _normalize_lon_to_180(lon):
    lon = np.asarray(lon)
    if np.nanmax(lon) > 180.0:
        lon = ((lon + 180.0) % 360.0) - 180.0
    return lon

def _build_topo_interpolator(topo_nc_path):
    with Dataset(topo_nc_path) as ds:
        topo_lon = np.array(ds.variables["lon"][:], dtype=np.float64)
        topo_lat = np.array(ds.variables["lat"][:], dtype=np.float64)
        topo     = np.array(ds.variables["topo"][:], dtype=np.float32)  # (lat, lon)

    topo_lon = _normalize_lon_to_180(topo_lon)

    if topo_lon[1] < topo_lon[0]:
        topo_lon = topo_lon[::-1]
        topo = topo[:, ::-1]
    if topo_lat[1] < topo_lat[0]:
        topo_lat = topo_lat[::-1]
        topo = topo[::-1, :]

    def interp(lon_pts, lat_pts):
        lon_pts = _normalize_lon_to_180(np.asarray(lon_pts, dtype=np.float64))
        lat_pts = np.asarray(lat_pts, dtype=np.float64)

        ix = np.searchsorted(topo_lon, lon_pts, side="right") - 1
        iy = np.searchsorted(topo_lat, lat_pts, side="right") - 1

        ix = np.clip(ix, 0, topo_lon.size - 2)
        iy = np.clip(iy, 0, topo_lat.size - 2)

        x0 = topo_lon[ix];   x1 = topo_lon[ix + 1]
        y0 = topo_lat[iy];   y1 = topo_lat[iy + 1]

        with np.errstate(invalid="ignore", divide="ignore"):
            tx = (lon_pts - x0) / (x1 - x0)
            ty = (lat_pts - y0) / (y1 - y0)
        tx = np.clip(tx, 0.0, 1.0)
        ty = np.clip(ty, 0.0, 1.0)

        f00 = topo[iy,     ix    ]
        f10 = topo[iy,     ix + 1]
        f01 = topo[iy + 1, ix    ]
        f11 = topo[iy + 1, ix + 1]

        return ( (1.0 - tx) * (1.0 - ty) * f00
               + tx         * (1.0 - ty) * f10
               + (1.0 - tx) * ty         * f01
               + tx         * ty         * f11 ).astype(np.float32)

    return interp


def accumulate_bins(lon, lat, ke, lon_edges, lat_edges, sum_grid, sum2_grid, cnt_grid):
    ix = np.digitize(lon, lon_edges) - 1
    iy = np.digitize(lat, lat_edges) - 1
    valid = (ix >= 0) & (ix < sum_grid.shape[1]) & (iy >= 0) & (iy < sum_grid.shape[0]) & np.isfinite(ke)
    if not np.any(valid):
        return
    ix = ix[valid]; iy = iy[valid]; vals = ke[valid]
    np.add.at(sum_grid, (iy, ix), vals)
    np.add.at(sum2_grid, (iy, ix), vals*vals)
    np.add.at(cnt_grid, (iy, ix), 1)

# ---------- Lecture d’un fichier ----------
def read_ke_from_nc(nc_path, prefer_u="ugosa_filtered", prefer_v="vgosa_filtered",
                    fallback_u="ugosa", fallback_v="vgosa",topo_interp=None, min_depth=None):
    
    with Dataset(nc_path, "r") as ds:
        # 1) variables u,v
        for cu in (prefer_u, fallback_u):
            if cu in ds.variables:
                uvar = cu
                break
        else:
            return None  
        for cv in (prefer_v, fallback_v):
            if cv in ds.variables:
                vvar = cv
                break
        else:
            return None  

        # 2) coord lon/lat
        vlon = "longitude" if "longitude" in ds.variables else ("lon" if "lon" in ds.variables else None)
        vlat = "latitude"  if "latitude"  in ds.variables else ("lat" if "lat" in ds.variables else None)
        if vlon is None or vlat is None:
            return None

        lon = np.array(ds.variables[vlon][:], dtype="float64")
        lat = np.array(ds.variables[vlat][:], dtype="float64")
        lon = to_m180_180(lon)

        if lon.ndim == 1 and lat.ndim == 1:
            LON, LAT = np.meshgrid(lon, lat)
        else:
            LON, LAT = lon, lat

        U = ds.variables[uvar]
        V = ds.variables[vvar]
        U.set_auto_maskandscale(True)
        V.set_auto_maskandscale(True)

        def read_var_as_float_nan(var):
            arr = var[:]  
            if np.ma.isMaskedArray(arr):
                return arr.filled(np.nan).astype("float64")
            else:
                return np.array(arr, dtype="float64")

        # si dimension temps présente, on prend la première tranche
        u = read_var_as_float_nan(U[0, ...]) if U.ndim == 3 else read_var_as_float_nan(U)
        v = read_var_as_float_nan(V[0, ...]) if V.ndim == 3 else read_var_as_float_nan(V)

        # 4) harmonisation éventuelle d’unités -> m/s
        def to_m_per_s(arr, var):
            units = (getattr(var, "units", "") or "").lower().replace(" ", "")
            if units in ("cm/s", "cm.s-1", "cms-1", "cms^-1", "cm*s^-1"):
                return arr / 100.0
            return arr

        u = to_m_per_s(u, U)
        v = to_m_per_s(v, V)

        # 5) EKE 
        ke = 0.5 * (u*u + v*v).astype("float64")
        # filter if necessary

        ke = np.where((ke >= 0) & np.isfinite(ke) & (ke < 2.0), ke, np.nan)

        # 6) flatten + med mask
        lonf = np.asarray(LON, dtype="float64").ravel()
        latf = np.asarray(LAT, dtype="float64").ravel()
        kef  = ke.ravel()
    
        inside = contains_xy(ROI_POLY, lonf, latf)
        m = inside & np.isfinite(kef)
    
        if not np.any(m):
            return None
    
        if (topo_interp is not None) and (min_depth is not None):
            topo_vals = topo_interp(lonf[m], latf[m])   # shape (N,)
            deep_ok = np.isfinite(topo_vals) & (topo_vals <= float(min_depth))
            if not np.any(deep_ok):
                return None
            return lonf[m][deep_ok], latf[m][deep_ok], kef[m][deep_ok]
    
        return lonf[m], latf[m], kef[m]

# ---------- Plot ----------
def plot_map(lon_c, lat_c, ke_mean, title=None, savefig=None, show=False, cmap="viridis", vmin=None, vmax=None):
    fig = plt.figure(figsize=(10, 5), constrained_layout=True)
    ax = plt.axes(projection=ccrs.Mercator())
    land = cfeature.NaturalEarthFeature("physical", "land", "50m", edgecolor="black", facecolor="lightgray")
    ax.add_feature(land, zorder=0)
    ax.coastlines(resolution="50m", color="black", linewidth=0.6)
    ax.set_extent([-6, 36, 30, 46], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linewidth=0.1, alpha=0.5)
    gl.top_labels = False; gl.right_labels = False

    dlon = np.diff(lon_c).mean() if lon_c.size > 1 else 0.1
    dlat = np.diff(lat_c).mean() if lat_c.size > 1 else 0.1
    lon_edges = np.concatenate(([lon_c[0]-0.5*dlon], 0.5*(lon_c[:-1]+lon_c[1:]), [lon_c[-1]+0.5*dlon]))
    lat_edges = np.concatenate(([lat_c[0]-0.5*dlat], 0.5*(lat_c[:-1]+lat_c[1:]), [lat_c[-1]+0.5*dlat]))
    LON, LAT = np.meshgrid(lon_edges, lat_edges)

    pcm = ax.pcolormesh(LON, LAT, ke_mean, shading="auto", cmap=cmap,
                        vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    cbar = fig.colorbar(pcm, ax=ax, orientation="vertical", shrink=0.5, pad=0.02)
    cbar.set_label("Mean KE (m² s⁻²)")

    if title: ax.set_title(title)
    if savefig:
        Path(savefig).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savefig, dpi=300)
    if show: plt.show()
    else: plt.close(fig)

import numpy.fft as nfft

def _bin_area_km2(lat_centers, dlon_deg, dlat_deg):
    R = 6371.0  # km
    dlon = np.deg2rad(dlon_deg)
    dlat = np.deg2rad(dlat_deg)
    return (R*R * dlon * dlat * np.cos(np.deg2rad(lat_centers))).astype(np.float64)

def _estimate_Lx_Ly_ke(ke_mean, mask, dx_km, dy_km):
    X = ke_mean.copy()
    X[~mask] = np.nan
    mu = np.nanmean(X)
    if not np.isfinite(mu):
        return np.nan, np.nan
    X = X - mu
    X[np.isnan(X)] = 0.0
    W = mask.astype(np.float64)

    F_X = nfft.rfftn(X, axes=(0, 1))
    corr_num = nfft.irfftn(F_X * np.conj(F_X), s=X.shape, axes=(0, 1))
    
    F_W = nfft.rfftn(W, axes=(0, 1))
    corr_den = nfft.irfftn(F_W * np.conj(F_W), s=W.shape, axes=(0, 1))

    with np.errstate(invalid="ignore", divide="ignore"):
        R = corr_num / np.maximum(corr_den, 1e-12)

    R0 = R[0, 0]
    if not np.isfinite(R0) or R0 <= 0:
        return np.nan, np.nan
    R = R / R0

    Ry = R[:, 0]
    Rx = R[0, :]

    ny, nx = R.shape
    lags_x = np.arange(nx) * dx_km
    lags_y = np.arange(ny) * dy_km

    def _e_folding_length(Rline, lags):
        target = 1.0 / np.e
        idx = np.where(Rline <= target)[0]
        if idx.size > 0:
            i = idx[0]
            if i == 0:
                return lags[0]
            # interpolation linéaire locale
            x0, x1 = lags[i-1], lags[i]
            y0, y1 = Rline[i-1], Rline[i]
            if np.isfinite(y0) and np.isfinite(y1) and (y1 != y0):
                return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))
            else:
                return float(x1)
        pos = np.maximum(Rline, 0.0)
        return float(np.trapz(pos, lags) / pos[0]) if pos[0] > 0 else np.nan

    Lx = _e_folding_length(Rx, lags_x)
    Ly = _e_folding_length(Ry, lags_y)
    return (Lx if np.isfinite(Lx) else np.nan,
            Ly if np.isfinite(Ly) else np.nan)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="KE moyen bin-moyenné sur Méditerranée (sans Mer Noire / Gascogne).")
    ap.add_argument("--indir", required=True, help="Dossier avec les fichiers .nc SWOT L3 Expert déjà téléchargés")
    ap.add_argument("--pattern", default="SWOT_L3_LR_SSH_Expert_*.nc", help="Glob pattern des fichiers .nc")
    ap.add_argument("--res", type=float, default=0.02, help="Résolution des bins en degrés (ex: 0.1)")
    ap.add_argument("--min-count", type=int, default=1, help="Minimum observations per bin to keep it")
    ap.add_argument("--cmap", type=str, default="RdBu_r")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--savefig", type=str, default=None)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--topo-file", default=None,
                    help="Fichier NetCDF bathymétrique (ex: etopo2.nc) avec variables lon(lon), lat(lat), topo(lat,lon).")
    ap.add_argument("--topo", type=float, default=None,
                    help="Seuil de profondeur (m, négatif). Ex: -200 garde uniquement topo <= -200 m. Si None: pas de filtre.")

    args = ap.parse_args()

    indir = Path(args.indir)
    files = sorted(indir.glob(args.pattern))
    if not files:
        print(f"Aucun fichier .nc dans {indir} avec le motif {args.pattern}")
        return

    topo_interp = None
    if args.topo is not None:
        if args.topo_file is None:
            raise SystemExit("Vous avez spécifié --topo sans --topo-file : impossible de filtrer.")
        topo_interp = _build_topo_interpolator(args.topo_file)
        print(f"[topo] Filtre profondeur actif: topo <= {args.topo:.0f} m (fichier: {args.topo_file})")


    lon_edges, lat_edges, lon_centers, lat_centers = make_grid(res=args.res)
    ny, nx = lat_centers.size, lon_centers.size
    sum_grid = np.zeros((ny, nx), dtype="float64")
    cnt_grid = np.zeros((ny, nx), dtype="int64")
    sum2_grid = np.zeros((ny, nx), dtype="float64")  

    for f in tqdm(files, desc="Accumulate KE"):
        out = read_ke_from_nc(f,topo_interp=topo_interp, min_depth=args.topo)
        if out is None: 
            continue
        lon, lat, ke = out
        accumulate_bins(lon, lat, ke, lon_edges, lat_edges, sum_grid,sum2_grid, cnt_grid)

    with np.errstate(invalid="ignore", divide="ignore"):
        ke_mean = sum_grid / cnt_grid
        ke_mean[cnt_grid < args.min_count] = np.nan  # exclude empty/under-sampled bins from the map
    finite_vals = ke_mean[np.isfinite(ke_mean)]
    if finite_vals.size > 0:
        vmax = np.nanmax(finite_vals)
        v99  = np.nanpercentile(finite_vals, 99.9)
        print(f"[CHECK] max(KE_mean)={vmax:.3e}, p99.9={v99:.3e}")
        if vmax > 1e1:
            outliers = np.argwhere(ke_mean > 1e1)
            print(f"[WARN] valeurs KE aberrantes: {outliers.shape[0]} bins > 10 m²/s²")
        
    # ---- Regional stats (area-weighted by bin; exclude NaNs) ----
    mask = np.isfinite(ke_mean) & (cnt_grid >= args.min_count)
    
    weights_line = np.cos(np.deg2rad(lat_centers)).astype(np.float64)  # shape (ny,)
    Wb = (weights_line[:, None] * mask.astype(np.float64))             # shape (ny, nx)
    
    den = np.nansum(Wb)
    if not np.isfinite(den) or den <= 0:
        raise RuntimeError("Aucun bin valide pour le calcul régional (den <= 0).")
    
    num = np.nansum(ke_mean * Wb)
    mu = num / den
    
    var_area = np.nansum(((ke_mean - mu)**2) * Wb) / den
    var_area = max(var_area, 0.0)          

    w_flat = Wb[mask]                      
    sumw   = float(np.nansum(w_flat))
    sumw2  = float(np.nansum(np.square(w_flat)))
    
    dlon = np.diff(lon_centers).mean() if lon_centers.size > 1 else args.res
    dlat = np.diff(lat_centers).mean() if lat_centers.size > 1 else args.res
    
    R_earth = 6371.0
    dlon_rad = np.deg2rad(dlon); dlat_rad = np.deg2rad(dlat)
    cosphi = np.cos(np.deg2rad(lat_centers))
    dx_line_km = R_earth * dlon_rad * cosphi           # (ny,)
    dy_km       = R_earth * dlat_rad
    
    line_area = _bin_area_km2(lat_centers, dlon, dlat) # (ny,)
    with np.errstate(invalid="ignore"):
        dx_km = np.nansum(dx_line_km * line_area) / np.nansum(line_area)
    if not np.isfinite(dx_km) or dx_km <= 0:
        dx_km = R_earth * dlon_rad * np.cos(np.deg2rad(np.nanmean(lat_centers)))
    
    Lx_km, Ly_km = _estimate_Lx_Ly_ke(ke_mean, mask, dx_km, dy_km)
    
    A_km2 = float(np.nansum(line_area[:, None] * mask))
    
    # Neff 
    if np.isfinite(Lx_km) and np.isfinite(Ly_km) and (Lx_km > 0) and (Ly_km > 0) and (A_km2 > 0):
        Neff = A_km2 / (2.0 * np.pi * Lx_km * Ly_km)
        Neff = max(Neff, 1.0)
        method = "spatial-corr"
    else:
        Neff = (sumw * sumw) / sumw2 if sumw2 > 0 else 1.0
        method = "independent-fallback"
    
    # Neff = min(Neff, float(mask.sum()))
    
    se_mean = std_area / np.sqrt(Neff)
    print(f"[SE] method={method} Lx≈{Lx_km:.1f} km Ly≈{Ly_km:.1f} km A≈{A_km2:.2e} km² Neff≈{Neff:.1f}")

    print(f"[DEBUG] valid_bins={mask.sum()}, sumw={sumw:.3e}, sumw2={sumw2:.3e}, Neff≈{Neff:.0f}")
    print(f"[STATS] mean={mu:.6e} std={std_area:.6e} se={se_mean:.6e} valid_bins={mask.sum()}")

    # Plot
    vmin = args.vmin if args.vmin is not None else np.nanpercentile(ke_mean, 2)
    vmax = args.vmax if args.vmax is not None else np.nanpercentile(ke_mean, 97)
    plot_map(lon_centers, lat_centers, ke_mean, 
             title="SWOT L3 Expert — Mean KE (Mediterranean, masked)",
             savefig=args.savefig, show=args.show, cmap=args.cmap, vmin=vmin, vmax=vmax)

if __name__ == "__main__":
    main()
