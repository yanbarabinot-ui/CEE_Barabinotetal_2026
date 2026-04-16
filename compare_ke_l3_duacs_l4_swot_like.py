from __future__ import annotations

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 10:45:56 2025

@author: yan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare L3 cycles vs Copernicus DUACS L4 (SEALEVEL_EUR_PHY_L4_MY_008_068) MKE over Mediterranean.

For each L3 cycle folder:
  - infer date range [d0,d1] from L3 filenames (YYYYMMDD in name)
  - download DUACS Europe L4 MY daily files covering [d0,d1] (subset Med bbox), via copernicusmarine
  - read ugosa/vgosa on the native DUACS grid and compute daily KE=0.5*(u^2+v^2)
  - average DUACS EKE over the cycle
  - optionally project & apply the L3-coverage mask (from L3 swaths) to the DUACS map (supersampling)
  - compute area-weighted regional mean EKE, std, and standard error (with correlation)
  - write a CSV (one line per cycle); optional: save EKE maps
"""

import re
import sys
import csv
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import copernicusmarine as cm  

import shapely
from shapely.geometry import box
from shapely import contains_xy

import matplotlib.pyplot as plt
import numpy.fft as nfft

# ----------------- Config DUACS product -----------------
PRODUCT_ID = "SEALEVEL_EUR_PHY_L4_MY_008_068"
FN_DUACS_DATE = re.compile(r"(\d{8})")

# ----------------- Mediterranean ROI -----------------
def build_med_mask_poly():
    med_bbox   = box(-6, 30, 36, 46)
    black_sea  = box(27, 41, 42, 47)
    bay_biscay = box(-6, 43, -1, 46)
    roi = shapely.difference(med_bbox, shapely.unary_union([black_sea, bay_biscay]))
    return shapely.buffer(roi, 0)

ROI_POLY = build_med_mask_poly()

def to_m180_180(lon):
    return ((lon + 180) % 360) - 180

# ----------------- L3 cycles (date range & optional mask) -----------------
FN_L3_DATE = re.compile(r"_(\d{8})T\d{6}_")  # ..._YYYYMMDDTHHMMSS_
def l3_cycle_date_range(l3_dir: Path) -> Tuple[date, date]:
    days = []
    for p in l3_dir.glob("SWOT_L3_LR_SSH_Expert_*.nc"):
        m = FN_L3_DATE.search(p.name)
        if m:
            d = datetime.strptime(m.group(1), "%Y%m%d").date()
            days.append(d)
    if not days:
        raise RuntimeError(f"No dates found in {l3_dir}")
    return min(days), max(days)

def make_grid(lon_min=-6, lon_max=36, lat_min=30, lat_max=46, res=0.02):
    lon_edges = np.arange(lon_min, lon_max + 1e-12, res)
    lat_edges = np.arange(lat_min, lat_max + 1e-12, res)
    lon_centers = 0.5*(lon_edges[:-1] + lon_edges[1:])
    lat_centers = 0.5*(lat_edges[:-1] + lat_edges[1:])
    return lon_edges, lat_edges, lon_centers, lat_centers

def accumulate_bins(lon, lat, val, lon_edges, lat_edges, sum_grid, cnt_grid):
    ix = np.digitize(lon, lon_edges) - 1
    iy = np.digitize(lat, lat_edges) - 1
    valid = (ix >= 0) & (ix < sum_grid.shape[1]) & (iy >= 0) & (iy < sum_grid.shape[0]) & np.isfinite(val)
    if not np.any(valid):
        return
    np.add.at(sum_grid, (iy[valid], ix[valid]), val[valid])
    np.add.at(cnt_grid,  (iy[valid], ix[valid]), 1)

def build_l3_coverage_mask_on_fine(l3_cycle_dir: Path, res=0.02, min_count=1):
    lon_edges, lat_edges, lon_c, lat_c = make_grid(res=res)
    ny, nx = lat_c.size, lon_c.size
    cnt_L3 = np.zeros((ny, nx), dtype="int64")
    from netCDF4 import Dataset as NCDS
    for f in l3_cycle_dir.glob("SWOT_L3_LR_SSH_Expert_*.nc"):
        try:
            with NCDS(f, "r") as ds:
                vlon = "longitude" if "longitude" in ds.variables else ("lon" if "lon" in ds.variables else None)
                vlat = "latitude"  if "latitude"  in ds.variables else ("lat" if "lat" in ds.variables else None)
                if vlon is None or vlat is None: 
                    continue
                lon = to_m180_180(np.array(ds.variables[vlon][:], dtype="float64"))
                lat = np.array(ds.variables[vlat][:], dtype="float64")
                # Option: base mask on valid SSH if available
                valid = None
                for vv in ("ssha_filtered","ssha"):
                    if vv in ds.variables:
                        arr = np.array(ds.variables[vv][:], dtype="float64")
                        valid = np.isfinite(arr) if valid is None else (valid & np.isfinite(arr))
                        break
                if lon.ndim == 1 and lat.ndim == 1:
                    LON, LAT = np.meshgrid(lon, lat)
                else:
                    LON, LAT = lon, lat
                lonf = np.asarray(LON).ravel(); latf = np.asarray(LAT).ravel()
                if valid is None:
                    valf = np.ones_like(lonf, dtype=bool)
                else:
                    valf = np.asarray(valid).ravel()
                finite = np.isfinite(lonf) & np.isfinite(latf) & valf
                inside = np.zeros_like(finite, dtype=bool)
                inside[finite] = contains_xy(ROI_POLY, lonf[finite], latf[finite])
                good = inside & valf
                if np.any(good):
                    accumulate_bins(lonf[good], latf[good], np.ones(good.sum(), dtype=float),
                                    lon_edges, lat_edges, np.zeros((ny,nx), float), cnt_L3)
        except OSError:
            continue
    min_mask = cnt_L3 >= min_count
    return lon_c, lat_c, min_mask

# Build the L3 daily maps
def build_l3_daily_mask(
    l3_files_for_day: list[Path],
    lon_edges: np.ndarray,
    lat_edges: np.ndarray,
    ny: int,
    nx: int,
    min_count: int = 1,
) -> np.ndarray:

    if not l3_files_for_day:
        return np.zeros((ny, nx), dtype=bool)

    from netCDF4 import Dataset as NCDS

    cnt_L3 = np.zeros((ny, nx), dtype="int64")
    dummy_sum = np.zeros((ny, nx), dtype="float64")

    for f in l3_files_for_day:
        try:
            with NCDS(f, "r") as ds:
                vlon = "longitude" if "longitude" in ds.variables else ("lon" if "lon" in ds.variables else None)
                vlat = "latitude"  if "latitude"  in ds.variables else ("lat" if "lat" in ds.variables else None)
                if vlon is None or vlat is None:
                    continue

                lon = to_m180_180(np.array(ds.variables[vlon][:], dtype="float64"))
                lat = np.array(ds.variables[vlat][:], dtype="float64")

                # masque de validité L3 (qualité + SSH)
                valid = None
                if "quality_flag" in ds.variables:
                    q = np.array(ds.variables["quality_flag"][:])
                    valid = (q == 0)

                cand_ssh = ("ssha_filtered", "ssha")
                for vv in cand_ssh:
                    if vv in ds.variables:
                        arr = np.array(ds.variables[vv][:], dtype="float64")
                        valid = np.isfinite(arr) if valid is None else (valid & np.isfinite(arr))
                        break

                if valid is None:
                    valid = np.ones_like(lon, dtype=bool)

                # vectorisation
                if lon.ndim == 1 and lat.ndim == 1:
                    LON, LAT = np.meshgrid(lon, lat)
                    VALID = valid
                else:
                    LON, LAT = lon, lat
                    VALID = valid

                lonf = np.asarray(LON).ravel()
                latf = np.asarray(LAT).ravel()
                valf = np.asarray(VALID).ravel()

                finite = np.isfinite(lonf) & np.isfinite(latf) & valf
                inside = np.zeros_like(finite, dtype=bool)
                inside[finite] = contains_xy(ROI_POLY, lonf[finite], latf[finite])
                good = inside & valf

                if np.any(good):
                    ones = np.ones(good.sum(), dtype=float)
                    accumulate_bins(
                        lonf[good], latf[good], ones,
                        lon_edges, lat_edges,
                        dummy_sum, cnt_L3
                    )
        except OSError:
            continue

    return (cnt_L3 >= min_count)


# --------- Project L3 fine mask -> DUACS (supersampling) ----------
def project_mask_supersampled_to_L4(
    lon_c_fine, lat_c_fine, mask_fine,
    lon_L4_1d, lat_L4_1d,
    k=6, frac_thresh=0.9
):
    """
    Project a fine (0/1) mask onto a coarser regular grid (DUACS) by supersampling.
    A DUACS cell is True if the fraction of k×k subpoints falling in mask_fine is >= frac_thresh.
    """
    nxL4, nyL4 = lon_L4_1d.size, lat_L4_1d.size
    dlon = np.diff(lon_L4_1d).mean()
    dlat = np.diff(lat_L4_1d).mean()

    tx = (np.arange(k) + 0.5) / k
    ty = (np.arange(k) + 0.5) / k
    TX, TY = np.meshgrid(tx, ty)
    sub_dx = dlon * (TX.ravel() - 0.5)
    sub_dy = dlat * (TY.ravel() - 0.5)

    LONc, LATc = np.meshgrid(lon_L4_1d, lat_L4_1d)
    sub_lon = (LONc[..., None] + sub_dx).reshape(-1)
    sub_lat = (LATc[..., None] + sub_dy).reshape(-1)

    # fine grid edges
    dlon_f = np.diff(lon_c_fine).mean()
    dlat_f = np.diff(lat_c_fine).mean()
    lon_edges_f = np.concatenate(([lon_c_fine[0] - 0.5*dlon_f],
                                  0.5*(lon_c_fine[:-1] + lon_c_fine[1:]),
                                  [lon_c_fine[-1] + 0.5*dlon_f]))
    lat_edges_f = np.concatenate(([lat_c_fine[0] - 0.5*dlat_f],
                                  0.5*(lat_c_fine[:-1] + lat_c_fine[1:]),
                                  [lat_c_fine[-1] + 0.5*dlat_f]))

    ix_f = np.digitize(sub_lon, lon_edges_f) - 1
    iy_f = np.digitize(sub_lat, lat_edges_f) - 1

    valid = (ix_f >= 0) & (ix_f < lon_c_fine.size) & (iy_f >= 0) & (iy_f < lat_c_fine.size)
    vals = np.zeros(ix_f.size, dtype=np.int8)
    vals[valid] = mask_fine[iy_f[valid], ix_f[valid]].astype(np.int8)

    num = np.zeros(nyL4*nxL4, dtype=np.int32)
    den = np.zeros_like(num)
    cell_ids = np.repeat(np.arange(nyL4*nxL4), k*k)
    np.add.at(num, cell_ids, vals)
    np.add.at(den, cell_ids, valid.astype(np.int8))

    frac = np.zeros_like(num, dtype=float)
    ok = den > 0
    frac[ok] = num[ok] / den[ok]
    return (frac.reshape(nyL4, nxL4) >= frac_thresh)

# ----------------- Area-weighted stats -----------------
def area_weighted_stats(field, lat1d):
    field = np.asarray(field, dtype=float)
    lat1d = np.asarray(lat1d, dtype=float)
    if field.ndim != 2 or lat1d.ndim != 1 or lat1d.size != field.shape[0]:
        raise ValueError("bad shapes")
    Wline = np.cos(np.deg2rad(lat1d))
    W = np.broadcast_to(Wline[:, None], field.shape)
    m = np.isfinite(field)
    if not np.any(m):
        return dict(mean=np.nan, std=np.nan, se=np.nan, n=0)
    Fm = np.where(m, field, 0.0)
    Wm = np.where(m, W, 0.0)
    den = Wm.sum()
    if den <= 0:
        return dict(mean=np.nan, std=np.nan, se=np.nan, n=0)
    mu = (Fm * Wm).sum() / den
    var = ((np.where(m, field - mu, 0.0) ** 2) * Wm).sum() / den
    std = float(np.sqrt(var) if var > 0 else 0.0)
    # Nombre de bins valides (non pondéré) pour l'erreur-type
    n = int(m.sum())
    se = std / np.sqrt(n) if n > 1 else np.nan
    return dict(mean=float(mu), std=std, se=se, n=n)

def _bin_area_km2_lines(lat1d: np.ndarray, dlon_deg: float, dlat_deg: float) -> np.ndarray:
    """Aire par ligne: R^2 * dlon * dlat * cos(phi) (km^2). Renvoie (ny,)."""
    R = 6371.0  # km
    return (R*R * np.deg2rad(dlon_deg) * np.deg2rad(dlat_deg) * np.cos(np.deg2rad(lat1d))).astype(np.float64)

def _estimate_Lx_Ly_ke(field2d: np.ndarray, mask2d: np.ndarray, dx_km: float, dy_km: float) -> tuple[float, float]:
    """
    Estime Lx, Ly (km) via l'autocorrélation 2D corrigée du masque.
    field2d: carte (NaN autorisés)
    mask2d : bool (True = valide) de même shape
    """
    X = field2d.copy()
    X[~mask2d] = np.nan
    mu = np.nanmean(X)
    if not np.isfinite(mu):
        return np.nan, np.nan
    X = X - mu
    X[np.isnan(X)] = 0.0
    W = mask2d.astype(np.float64)

    # autocorr avec correction de masque (compat NumPy>=2.0)
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

    Rx = R[0, :]
    Ry = R[:, 0]
    lags_x = np.arange(Rx.size) * dx_km
    lags_y = np.arange(Ry.size) * dy_km

    def _e_fold(Rline, lags):
        target = 1.0/np.e
        idx = np.where(Rline <= target)[0]
        if idx.size:
            i = idx[0]
            if i == 0:
                return lags[0]
            x0, x1 = lags[i-1], lags[i]
            y0, y1 = Rline[i-1], Rline[i]
            if np.isfinite(y0) and np.isfinite(y1) and (y1 != y0):
                return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))
            return float(x1)
        # fallback: intégrale positive
        pos = np.maximum(Rline, 0.0)
        return float(np.trapz(pos, lags) / pos[0]) if pos[0] > 0 else np.nan

    Lx = _e_fold(Rx, lags_x)
    Ly = _e_fold(Ry, lags_y)
    return (Lx if np.isfinite(Lx) else np.nan, Ly if np.isfinite(Ly) else np.nan)

def spatial_standard_error_with_corr_duacs(field2d: np.ndarray,
                                           lon1d: np.ndarray,
                                           lat1d: np.ndarray) -> tuple[float, float, float, float, float, float, str]:
    """
    SE corrigée de la corrélation spatiale pour une carte DUACS:
      - field2d: EKE moyenne (NaN hors zone)
      - lon1d, lat1d: axes 1D réguliers de la grille DUACS
    Retourne: (mu, std_area, se_corr, Neff, Lx_km, Ly_km, method)
    """
    field2d = np.asarray(field2d, dtype=float)
    lon1d = np.asarray(lon1d, dtype=float)
    lat1d = np.asarray(lat1d, dtype=float)

    mask = np.isfinite(field2d)
    if not np.any(mask):
        return np.nan, np.nan, np.nan, 1.0, np.nan, np.nan, "no-data"

    # poids surfaciques ~ cos(phi)
    Wline = np.cos(np.deg2rad(lat1d)).astype(np.float64)
    Wb = (Wline[:, None] * mask).astype(np.float64)
    den = np.nansum(Wb)
    if not np.isfinite(den) or den <= 0:
        return np.nan, np.nan, np.nan, 1.0, np.nan, np.nan, "no-data"

    mu = np.nansum(field2d * Wb) / den
    var = np.nansum(((field2d - mu)**2) * Wb) / den
    std_area = float(np.sqrt(max(var, 0.0)))

    # métrique grille
    dlon = np.diff(lon1d).mean() if lon1d.size > 1 else 0.0625
    dlat = np.diff(lat1d).mean() if lat1d.size > 1 else 0.0625
    R_earth = 6371.0
    dx_line_km = R_earth * np.deg2rad(dlon) * np.cos(np.deg2rad(lat1d))
    dy_km = R_earth * np.deg2rad(dlat)
    line_area = _bin_area_km2_lines(lat1d, dlon, dlat)

    with np.errstate(invalid="ignore"):
        dx_km = np.nansum(dx_line_km * line_area) / np.nansum(line_area)
    if not np.isfinite(dx_km) or dx_km <= 0:
        dx_km = R_earth * np.deg2rad(dlon) * np.cos(np.deg2rad(np.nanmean(lat1d)))

    # Lx, Ly
    Lx_km, Ly_km = _estimate_Lx_Ly_ke(field2d, mask, dx_km, dy_km)

    # Aire valide (km²)
    A_km2 = float(np.nansum(line_area[:, None] * mask))

    # Neff corrélé
    if np.isfinite(Lx_km) and np.isfinite(Ly_km) and (Lx_km > 0) and (Ly_km > 0) and (A_km2 > 0):
        Neff = A_km2 / (2.0 * np.pi * Lx_km * Ly_km)
        Neff = max(Neff, 1.0)
        method = "spatial-corr"
    else:
        w_flat = Wb[mask]
        sumw = float(np.nansum(w_flat))
        sumw2 = float(np.nansum(np.square(w_flat)))
        Neff = (sumw * sumw) / sumw2 if sumw2 > 0 else 1.0
        method = "independent-fallback"

    se_corr = std_area / np.sqrt(Neff)
    return float(mu), float(std_area), float(se_corr), float(Neff), float(Lx_km), float(Ly_km), method

# ----------------- Plot helper -----------------
def plot_map(lon1d, lat1d, field, title, savefig=None, cmap="RdBu_r", vmin=None, vmax=None):
    fig = plt.figure(figsize=(9, 6), constrained_layout=True)
    ax = plt.axes(projection=ccrs.PlateCarree())
    land = cfeature.NaturalEarthFeature("physical","land","50m", edgecolor="black", facecolor="lightgray")
    ax.add_feature(land); ax.coastlines(resolution="50m", color="black", linewidth=0.6)
    ax.set_extent([-6,36,30,46], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False; gl.right_labels = False

    dlon = np.diff(lon1d).mean() if lon1d.size>1 else 0.1
    dlat = np.diff(lat1d).mean() if lat1d.size>1 else 0.1
    lon_edges = np.concatenate(([lon1d[0]-0.5*dlon], 0.5*(lon1d[:-1]+lon1d[1:]), [lon1d[-1]+0.5*dlon]))
    lat_edges = np.concatenate(([lat1d[0]-0.5*dlat], 0.5*(lat1d[:-1]+lat1d[1:]), [lat1d[-1]+0.5*dlat]))
    LON, LAT = np.meshgrid(lon_edges, lat_edges)
    pcm = ax.pcolormesh(LON, LAT, field, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(pcm, ax=ax, shrink=0.5); cbar.set_label("KE (m² s⁻²)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_title(title)
    if savefig:
        Path(savefig).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savefig, dpi=170)
    plt.close(fig)

# ----------------- DUACS download & read -----------------
def month_range(d0: date, d1: date) -> List[Tuple[int,int]]:
    """Return list of (year,month) covering [d0,d1]."""
    out = []
    y, m = d0.year, d0.month
    while True:
        out.append((y,m))
        if (y==d1.year and m==d1.month): break
        m += 1
        if m>12: m=1; y+=1
    return out

DUACS_DAILY_DATASET_ID = "cmems_obs-sl_eur_phy-ssh_my_allsat-l4-duacs-0.0625deg_P1D"

# Download daily DUACS files (year, month) 
def duacs_download_month(year: int, month: int, cache_root: Path, force: bool = False):

    out_flat = cache_root / f"{year:04d}" / f"{month:02d}"
    out_flat.mkdir(parents=True, exist_ok=True)

    if not force and any(out_flat.glob("*.nc")):
        return

    rawdir = cache_root / "_RAW" / f"{year:04d}{month:02d}"
    rawdir.mkdir(parents=True, exist_ok=True)

    month_filter = f"*{year:04d}/{month:02d}/*"

    try:
        cm.get(
            dataset_id=DUACS_DAILY_DATASET_ID,
            filter=month_filter,
            output_directory=str(rawdir),
            no_directories=False,   
            skip_existing=True      # évite re-téléchargements au prochain run
        )
    except Exception as e:
        print(f"[WARN] cm.get failed for {year}-{month:02d}: {e}")

    for nc in rawdir.rglob("*.nc"):
        target = out_flat / nc.name
        if not target.exists():
            try:
                nc.replace(target)  # move
            except Exception:
                import shutil
                shutil.copy2(nc, target)
                try:
                    nc.unlink()
                except Exception:
                    pass

    try:
        for p in sorted(rawdir.rglob("*"), reverse=True):
            if p.is_dir():
                try: p.rmdir()
                except OSError: pass
        rawdir.rmdir()
    except Exception:
        pass

# Return (lon1d, lat1d, ugos2d, vgos2d, day) masked outside Med, robust to masked/packed data.
def read_duacs_daily(nc_path: Path):
    try:
        with Dataset(nc_path, "r") as ds:
            ds.set_auto_maskandscale(True) 

            # coords
            vlon = "longitude" if "longitude" in ds.variables else ("lon" if "lon" in ds.variables else None)
            vlat = "latitude"  if "latitude"  in ds.variables else ("lat" if "lat" in ds.variables else None)
            if vlon is None or vlat is None:
                return None
            lon = np.array(ds.variables[vlon][:], dtype="float64")
            lat = np.array(ds.variables[vlat][:], dtype="float64")
            lon = to_m180_180(lon)

            # variables u,v 
            cand_u = ["ugosa"]
            cand_v = ["vgosa"]
            Uname = next((v for v in cand_u if v in ds.variables), None)
            Vname = next((v for v in cand_v if v in ds.variables), None)
            if Uname is None or Vname is None:
                return None
            U = ds.variables[Uname]; V = ds.variables[Vname]

            # lecture "masked array" → remplissage NaN
            u_raw = U[0, ...] if U.ndim==3 else U[:]
            v_raw = V[0, ...] if V.ndim==3 else V[:]
            u = np.array(np.ma.filled(u_raw, np.nan), dtype="float64")
            v = np.array(np.ma.filled(v_raw, np.nan), dtype="float64")

            # conversion unités si besoin (ex. 'cm/s' → m/s)
            for arr, Var in ((u, U), (v, V)):
                units = getattr(Var, "units", "").lower()
                if "cm/s" in units or "cm s-1" in units or "cm s^-1" in units:
                    arr /= 100.0  # in-place ok car arr est une vue locale
                # si déjà 'm/s' → rien à faire

            day = None
            tvar = next((nm for nm in ("time","day","t") if nm in ds.variables), None)
            if tvar:
                t = np.array(ds.variables[tvar][:]).ravel()
                if t.size > 0:
                    try:
                        units = getattr(ds.variables[tvar], "units", "")
                        calendar = getattr(ds.variables[tvar], "calendar", "standard")
                        import cftime
                        dts = cftime.num2date(t, units=units, calendar=calendar)
                        dt0 = dts[0] if np.ndim(dts) else dts
                        # reconstruit un datetime.date Python standard
                        day = date(int(dt0.year), int(dt0.month), int(dt0.day))
                    except Exception:
                        day = None

            # fallback si la date n'a pas été obtenue via la variable temps
            if day is None:
                m = FN_DUACS_DATE.search(nc_path.name)
                if m:
                    try:
                        day = datetime.strptime(m.group(1), "%Y%m%d").date()
                    except Exception:
                        day = None


            # masque Méditerranée
            if lon.ndim==1 and lat.ndim==1:
                LON, LAT = np.meshgrid(lon, lat)
            else:
                LON, LAT = lon, lat
                lon = lon[0,:] if lon.ndim==2 else lon
                lat = lat[:,0] if lat.ndim==2 else lat
            flat = np.isfinite(LON.ravel()) & np.isfinite(LAT.ravel())
            inside = np.zeros(LON.size, dtype=bool)
            inside[flat] = contains_xy(ROI_POLY, LON.ravel()[flat], LAT.ravel()[flat])
            inside = inside.reshape(LON.shape)

            u = np.where(inside, u, np.nan)
            v = np.where(inside, v, np.nan)

            return lon, lat, u, v, day
    except Exception:
        return None

# ----------------- Cycle processing -----------------
def process_cycle_duacs(cycle_dir_l3: Path, duacs_cache_root: Path,
                        apply_l3_mask: bool,
                        l3_mask_res: float = 0.025,
                        l3_mask_min_count: int = 1,
                        mask_supersample_k: int = 6,
                        mask_frac_thresh: float = 0.9,
                        plot_map_png: Optional[Path] = None) -> Dict[str, Any]:
    """
    Si apply_l3_mask == False :
      - EKE DUACS "plein champ" sur la Méditerranée (hors ROI non-Med).

    Si apply_l3_mask == True (mode SWOT-like) :
      - DUACS = "vrai océan".
      - Pour chaque jour DUACS :
          * on cherche les fichiers L3 du même jour,
          * on construit un masque L3 JOURNALIER sur une grille fine,
          * on projette ce masque sur la grille DUACS,
          * on accumule KE DUACS seulement là où L3 a vu ce jour-là.
      - À la fin : carte EKE DUACS "SWOT-like" sur le cycle.
    """
    # 1) date window from L3
    d0, d1 = l3_cycle_date_range(cycle_dir_l3)
    months = month_range(d0, d1)

    # 2) download DUACS for these months (ugos/vgos only)
    for (yy, mm) in months:
        duacs_download_month(yy, mm, duacs_cache_root, force=False)

    # 3) Préparation L3 si masque SWOT-like demandé
    if apply_l3_mask:
        # grille fine pour les masques journaliers
        lon_edges_f, lat_edges_f, lon_c_f, lat_c_f = make_grid(res=l3_mask_res)
        ny_fine, nx_fine = lat_c_f.size, lon_c_f.size

        # associe chaque fichier L3 à sa date (d'après le nom)
        day_to_l3files: Dict[date, list[Path]] = {}
        for f in cycle_dir_l3.glob("SWOT_L3_LR_SSH_Expert_*.nc"):
            m = FN_L3_DATE.search(f.name)
            if not m:
                continue
            d = datetime.strptime(m.group(1), "%Y%m%d").date()
            day_to_l3files.setdefault(d, []).append(f)

        # caches masques (pour éviter de recalculer pour un même jour)
        daily_mask_fine_cache: Dict[date, np.ndarray] = {}
        daily_mask_on_DU_cache: Dict[date, np.ndarray] = {}
    else:
        day_to_l3files = {}
        daily_mask_fine_cache = {}
        daily_mask_on_DU_cache = {}

    # 4) lecture DUACS journalière + accumulation KE
    lonDU = latDU = None
    sum_ke = cnt_ke = None
    l4_files = 0

    for (yy, mm) in months:
        month_dir = duacs_cache_root / f"{yy:04d}" / f"{mm:02d}"
        if not month_dir.is_dir():
            continue

        for nc in sorted(month_dir.glob("*.nc")):
            out = read_duacs_daily(nc)
            if out is None:
                continue
            lon1d, lat1d, u2d, v2d, day = out

            # keep only days within window
            if day is not None and (day < d0 or day > d1):
                continue

            ke = 0.5*(u2d*u2d + v2d*v2d)

            if lonDU is None:
                lonDU, latDU = lon1d, lat1d
                ny, nx = latDU.size, lonDU.size
                sum_ke = np.zeros((ny,nx), dtype="float64")
                cnt_ke = np.zeros((ny,nx), dtype="int64")

            # sécurité grille
            if (len(lon1d) != len(lonDU)) or (len(lat1d) != len(latDU)) \
               or not np.allclose(lon1d, lonDU) or not np.allclose(lat1d, latDU):
                continue

            # ---------- CAS SANS MASQUE L3 : même comportement qu'avant ----------
            if not apply_l3_mask:
                valid = np.isfinite(ke)
                if not np.any(valid):
                    continue
                sum_ke[valid] += ke[valid]
                cnt_ke[valid] += 1
                l4_files += 1
                continue

            # ---------- CAS SWOT-like : masque L3 JOURNALIER ----------
            if day is None:
                # pas de date exploitable → on ne peut pas matcher avec L3
                continue

            l3_files_today = day_to_l3files.get(day, [])
            if not l3_files_today:
                # pas de passe L3 ce jour-là → on considère que SWOT n'a pas échantillonné DUACS ce jour
                continue

            # masque fin pour ce jour
            if day in daily_mask_fine_cache:
                mask_fine_day = daily_mask_fine_cache[day]
            else:
                mask_fine_day = build_l3_daily_mask(
                    l3_files_for_day=l3_files_today,
                    lon_edges=lon_edges_f,
                    lat_edges=lat_edges_f,
                    ny=ny_fine,
                    nx=nx_fine,
                    min_count=l3_mask_min_count,
                )
                daily_mask_fine_cache[day] = mask_fine_day

            if not np.any(mask_fine_day):
                continue

            # projection masque fin -> DUACS pour ce jour
            if day in daily_mask_on_DU_cache:
                mask_on_DU_day = daily_mask_on_DU_cache[day]
            else:
                mask_on_DU_day = project_mask_supersampled_to_L4(
                    lon_c_fine=lon_c_f, lat_c_fine=lat_c_f, mask_fine=mask_fine_day,
                    lon_L4_1d=lonDU, lat_L4_1d=latDU,
                    k=mask_supersample_k, frac_thresh=mask_frac_thresh
                )
                daily_mask_on_DU_cache[day] = mask_on_DU_day

            if not np.any(mask_on_DU_day):
                continue

            ke_masked = np.where(mask_on_DU_day, ke, np.nan)
            valid = np.isfinite(ke_masked)
            if not np.any(valid):
                continue

            sum_ke[valid] += ke_masked[valid]
            cnt_ke[valid] += 1
            l4_files += 1

    if sum_ke is None or np.nanmax(cnt_ke)==0:
        return {"cycle_dir": cycle_dir_l3.name, "date_min": d0.isoformat(), "date_max": d1.isoformat(),
                "l4_files": l4_files, "mean_ke": np.nan, "std_ke": np.nan, "se_ke": np.nan}

    ke_mean = np.full_like(sum_ke, np.nan, dtype=float)
    m = cnt_ke > 0
    ke_mean[m] = sum_ke[m] / cnt_ke[m]

    # stats (area-weighted)
    stats = area_weighted_stats(ke_mean, latDU)

    # SE corrigée corrélation spatiale
    mu_corr, std_corr, se_corr, Neff, Lx_km, Ly_km, method = spatial_standard_error_with_corr_duacs(
        ke_mean, lonDU, latDU
    )
    print(f"[SE] method={method} Neff≈{Neff:.1f} Lx≈{Lx_km:.1f} km Ly≈{Ly_km:.1f} km")

    if plot_map_png is not None:
        vmin = np.nanpercentile(ke_mean, 2) if np.isfinite(ke_mean).any() else None
        vmax = np.nanpercentile(ke_mean, 98) if np.isfinite(ke_mean).any() else None
        plot_map(lonDU, latDU, ke_mean,
                 f"DUACS L4 EKE (SWOT-like={apply_l3_mask})\n{cycle_dir_l3.name} [{d0}..{d1}]",
                 savefig=str(plot_map_png), vmin=vmin, vmax=vmax)

    return {"cycle_dir": cycle_dir_l3.name,
            "date_min": d0.isoformat(),
            "date_max": d1.isoformat(),
            "l4_files": l4_files,
            "mean_ke": stats["mean"],
            "std_ke": stats["std"],
            "se_ke": se_corr}

# ----------------- CLI -----------------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Compare L3 cycles vs DUACS L4 MKE over Mediterranean.")
    ap.add_argument("--l3_root", required=True, help="Folder containing L3 cycle subfolders (e.g., swot_l3_multi_cycles)")
    ap.add_argument("--cycles", type=str, default=None, help="Comma-separated cycle numbers (e.g., 1,2,3); default: all cycle_*")
    ap.add_argument("--duacs_cache", type=str, default="duacs_l4_cache", help="Where to cache DUACS daily files")
    ap.add_argument("--apply-l3-mask", action="store_true", help="Activate L3 coverage mask projection onto DUACS grid")
    ap.add_argument("--l3-mask-res", type=float, default=0.025, help="Fine grid resolution for L3 mask (deg)")
    ap.add_argument("--l3-mask-min-count", type=int, default=1, help="Min obs per fine bin for L3 mask")
    ap.add_argument("--mask-k", type=int, default=6, help="Supersampling factor k for L3->DUACS projection")
    ap.add_argument("--mask-frac", type=float, default=0.9, help="Fraction threshold for L3->DUACS projection")
    ap.add_argument("--savecsv", type=str, default="compare_ke_l3_duacs_l4_swot_like.csv")
    ap.add_argument("--plot-per-cycle", action="store_true", help="Save a EKE map per cycle")
    args = ap.parse_args()

    l3_root = Path(args.l3_root)
    if args.cycles:
        cycles = [int(x) for x in args.cycles.split(",")]
        cycle_dirs = [l3_root / f"cycle_{c:03d}" for c in cycles]
    else:
        cycle_dirs = sorted([p for p in l3_root.glob("cycle_*") if p.is_dir()])

    rows = []
    duacs_cache = Path(args.duacs_cache)

    for cdir in cycle_dirs:
        print(f"\n=== Processing {cdir.name} ===")
        map_png = (duacs_cache / cdir.name / f"mke_duacs_{cdir.name}.png") if args.plot_per_cycle else None
        stats = process_cycle_duacs(
            cdir, duacs_cache,
            apply_l3_mask=args.apply_l3_mask,
            l3_mask_res=args.l3_mask_res,
            l3_mask_min_count=args.l3_mask_min_count,
            mask_supersample_k=args.mask_k,
            mask_frac_thresh=args.mask_frac,
            plot_map_png=map_png
        )
        print(stats)
        rows.append({"cycle": cdir.name.split("_")[-1],
                     "date_min": stats.get("date_min"),
                     "date_max": stats.get("date_max"),
                     "l4_files": stats.get("l4_files", 0),
                     "mean_ke": stats.get("mean_ke"),
                     "std_ke": stats.get("std_ke"),
                     "se_ke": stats.get("se_ke")})

    # CSV
    csv_path = Path(args.savecsv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cycle","date_min","date_max","l4_files","mean_ke","std_ke","se_ke"])
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nSaved: {csv_path}")

if __name__ == "__main__":
    main()
