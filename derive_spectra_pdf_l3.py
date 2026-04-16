from __future__ import annotations

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Oct 16 14:37:44 2025

@author: yan
"""

"""
Derive SSH spectrum + PDFs (strain, normalized vorticity) per SWOT L3 cycle (Mediterranean)

Outputs (CSV, long format):
  - ssh_spectra.csv       : cycle,k_cpkm,E_ssh
  - pdf_strain.csv        : cycle,bin_center,pdf
  - pdf_vort_norm.csv     : cycle,bin_center,pdf

Optional figures per cycle:
  - map_strain_cXXX.png
  - map_vortnorm_cXXX.png

Usage:
  python derive_spectra_pdf_l3.py \
      --l3_root swot_l3_multi_cycles \
      --res 0.02 --min-count 3 \
      --area-weight \
      --savefigs

"""

import re
import csv
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import shapely
from shapely.geometry import box
from shapely import contains_xy

# ---------- Constants ----------
EARTH_RADIUS_M = 6371000.0
OMEGA = 7.292115e-5  # rad/s

# ---------- Mediterranean ROI ----------
def build_med_mask():
    med_bbox   = box(-6, 30, 36, 46)
    black_sea  = box(27, 41, 42, 47)
    bay_biscay = box(-6, 43, -1, 46)
    roi = shapely.difference(med_bbox, shapely.unary_union([black_sea, bay_biscay]))
    return shapely.buffer(roi, 0)

ROI_POLY = build_med_mask()

def to_m180_180(lon):
    return ((lon + 180) % 360) - 180

# ---------- Grid & binning ----------
def make_grid(lon_min=-6, lon_max=36, lat_min=30, lat_max=46, res=0.02):
    lon_edges = np.arange(lon_min, lon_max + 1e-12, res)
    lat_edges = np.arange(lat_min, lat_max + 1e-12, res)
    lon_centers = 0.5*(lon_edges[:-1] + lon_edges[1:])
    lat_centers = 0.5*(lat_edges[:-1] + lat_edges[1:])
    return lon_edges, lat_edges, lon_centers, lat_centers

def accumulate_bins(lon, lat, val, lon_edges, lat_edges, sum_grid, cnt_grid):
    ix = np.digitize(lon, lon_edges) - 1
    iy = np.digitize(lat, lat_edges) - 1
    H, W = sum_grid.shape
    valid = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H) & np.isfinite(val)
    if not np.any(valid):
        return
    np.add.at(sum_grid, (iy[valid], ix[valid]), val[valid])
    np.add.at(cnt_grid,  (iy[valid], ix[valid]), 1)

# ---------- Your helper functions (slightly wrapped) ----------
def row_angle_from_finite_ends(i, LON, LAT, return_degrees=False):
    lon_row = LON[i, :]; lat_row = LAT[i, :]
    finite = np.isfinite(lon_row) & np.isfinite(lat_row)
    if finite.sum() < 2:
        return np.nan
    idx = np.where(finite)[0]
    i1, i2 = idx[0], idx[-1]
    lon1, lat1 = lon_row[i1], lat_row[i1]
    lon2, lat2 = lon_row[i2], lat_row[i2]

    dlon_deg = (lon2 - lon1 + 180.0) % 360.0 - 180.0
    dphi = np.deg2rad(lat2 - lat1)
    dlmb = np.deg2rad(dlon_deg)
    phi_bar = np.deg2rad(0.5 * (lat1 + lat2))
    dx = EARTH_RADIUS_M * np.cos(phi_bar) * dlmb
    dy = EARTH_RADIUS_M * dphi
    ang = np.arctan2(dy, dx)
    if return_degrees: ang = np.rad2deg(ang)
    return ang


def grid_dx_dy(lon: np.ndarray, lat: np.ndarray):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lam = np.deg2rad(lon)
    phi = np.deg2rad(lat)
    cosphi = np.cos(phi)

    dlam_r = np.empty_like(lam); dphi_r = np.empty_like(phi)
    dlam_r[:, :-1] = lam[:, 1:] - lam[:, :-1]; dphi_r[:, :-1] = phi[:, 1:] - phi[:, :-1]
    dlam_r[:, -1]  = dlam_r[:, -2];            dphi_r[:, -1]  = dphi_r[:, -2]
    dx_right = EARTH_RADIUS_M * np.sqrt(dphi_r**2 + (cosphi * dlam_r)**2)

    dlam_d = np.empty_like(lam); dphi_d = np.empty_like(phi)
    dlam_d[:-1, :] = lam[1:, :] - lam[:-1, :]; dphi_d[:-1, :] = phi[1:, :] - phi[:-1, :]
    dlam_d[-1, :]  = dlam_d[-2, :];            dphi_d[-1, :]  = dphi_d[-2, :]
    dy_down = EARTH_RADIUS_M * np.sqrt(dphi_d**2 + (cosphi * dlam_d)**2)

    return dx_right, dy_down

def central_gradient(a: np.ndarray, dx_right: np.ndarray, dy_down: np.ndarray):
    a = np.asarray(a, dtype=float)
    dax = np.full_like(a, np.nan)
    day = np.full_like(a, np.nan)

    # dx gauche/droite
    dx_left  = np.empty_like(dx_right)
    dx_left[:, 1:] = dx_right[:, :-1]
    dx_left[:, 0]  = dx_right[:, 0]
    dx_sum = dx_left + dx_right

    # masque de validité
    A = np.isfinite(a)

    # ----- dérivée selon x -----
    m_c = A[:, 2:] & A[:, :-2] & np.isfinite(dx_sum[:, 1:-1]) & (dx_sum[:, 1:-1] > 0)
    dax[:, 1:-1][m_c] = (a[:, 2:][m_c] - a[:, :-2][m_c]) / dx_sum[:, 1:-1][m_c]

    m_f = A[:, 1] & A[:, 0] & np.isfinite(dx_right[:, 0]) & (dx_right[:, 0] > 0)
    dax[:, 0][m_f] = (a[:, 1][m_f] - a[:, 0][m_f]) / dx_right[:, 0][m_f]

    m_b = A[:, -1] & A[:, -2] & np.isfinite(dx_left[:, -1]) & (dx_left[:, -1] > 0)
    dax[:, -1][m_b] = (a[:, -1][m_b] - a[:, -2][m_b]) / dx_left[:, -1][m_b]

    # ----- dérivée selon y -----
    dy_up = np.empty_like(dy_down)
    dy_up[1:, :] = dy_down[:-1, :]
    dy_up[0,  :] = dy_down[0,  :]
    dy_sum = dy_up + dy_down

    m_cy = A[2:, :] & A[:-2, :] & np.isfinite(dy_sum[1:-1, :]) & (dy_sum[1:-1, :] > 0)
    day[1:-1, :][m_cy] = (a[2:, :][m_cy] - a[:-2, :][m_cy]) / dy_sum[1:-1, :][m_cy]

    m_fy = A[1, :] & A[0, :] & np.isfinite(dy_down[0, :]) & (dy_down[0, :] > 0)
    day[0, :][m_fy] = (a[1, :][m_fy] - a[0, :][m_fy]) / dy_down[0, :][m_fy]

    m_by = A[-1, :] & A[-2, :] & np.isfinite(dy_up[-1, :]) & (dy_up[-1, :] > 0)
    day[-1, :][m_by] = (a[-1, :][m_by] - a[-2, :][m_by]) / dy_up[-1, :][m_by]

    return dax, day


def projection_gradients(dudX, dudY, LON, LAT):
    nrows = dudX.shape[0]
    dudx = np.full_like(dudX, np.nan)
    dudy = np.full_like(dudX, np.nan)
    for i in range(nrows):
        ang = row_angle_from_finite_ends(i, LON, LAT)  # rad
        if not np.isfinite(ang):
            continue  # laisse NaN pour cette ligne
        c, s = np.cos(ang), np.sin(ang)
        dudx[i, :] = c * dudX[i, :] - s * dudY[i, :]
        dudy[i, :] = s * dudX[i, :] + c * dudY[i, :]
    return dudx, dudy

def omega_f(dvdx, dudy, f):
    return (dvdx - dudy) / f

def Strain(dudx, dudy, dvdx, dvdy):
    s_n = dudx - dvdy
    s_s = dudy + dvdx
    return np.sqrt(s_n**2 + s_s**2)

# ---------- Read needed fields (robust masked read) ----------
def read_vars_from_nc(nc_path, ssh_candidates=("ssha_filtered",),
                      u_candidates=("ugosa_filtered",),
                      v_candidates=("vgosa_filtered",)):
        
    with Dataset(nc_path, "r") as ds:

        # coords
        vlon = "longitude" if "longitude" in ds.variables else ("lon" if "lon" in ds.variables else None)
        vlat = "latitude"  if "latitude"  in ds.variables else ("lat" if "lat" in ds.variables else None)
        if vlon is None or vlat is None:
            return None
        lon = np.array(ds.variables[vlon][:], dtype="float64")
        lat = np.array(ds.variables[vlat][:], dtype="float64")
        lon = to_m180_180(lon)

        # choose variables
        ssh_var = next((nm for nm in ssh_candidates if nm in ds.variables), None)
        u_var   = next((nm for nm in u_candidates   if nm in ds.variables), None)
        v_var   = next((nm for nm in v_candidates   if nm in ds.variables), None)
        if ssh_var is None or u_var is None or v_var is None:
            return None

        SSH = ds.variables[ssh_var]
        U   = ds.variables[u_var]
        V   = ds.variables[v_var]

        # masked+scaled -> float with NaN
        SSH.set_auto_maskandscale(True); U.set_auto_maskandscale(True); V.set_auto_maskandscale(True)

        def read_nan(var):
            arr = var[:]
            return (arr.filled(np.nan) if np.ma.isMaskedArray(arr) else np.array(arr)).astype("float64")

        ssh = read_nan(SSH[0, ...]) if SSH.ndim == 3 else read_nan(SSH)
        u   = read_nan(U[0, ...])   if U.ndim   == 3 else read_nan(U)
        v   = read_nan(V[0, ...])   if V.ndim   == 3 else read_nan(V)

        # quick unit harmonization for velocities (cm/s -> m/s)
        def to_mps(arr, var):
            units = (getattr(var, "units", "") or "").lower().replace(" ", "")
            return arr/100.0 if units.startswith("cm/s") or units.replace(".","").startswith("cms-1") else arr
        u = to_mps(u, U); v = to_mps(v, V)

        # flatten helpers later need 2D lon/lat
        if lon.ndim == 1 and lat.ndim == 1:
            LON, LAT = np.meshgrid(lon, lat)
        else:
            LON, LAT = lon, lat

        # keep only Mediterranean footprint
        
        lonf = np.asarray(LON).ravel()
        latf = np.asarray(LAT).ravel()
        finite_xy = np.isfinite(lonf) & np.isfinite(latf)
        inside_flat = np.zeros_like(finite_xy, dtype=bool)
        inside_flat[finite_xy] = contains_xy(ROI_POLY, lonf[finite_xy], latf[finite_xy])
        inside = inside_flat.reshape(LON.shape)
        
        # masque
        ssh = np.where(inside, ssh, np.nan)
        u   = np.where(inside, u,   np.nan)
        v   = np.where(inside, v,   np.nan)
        return LON, LAT, ssh, u, v

from numpy.fft import rfft, rfftfreq

def welch_rowwise_psd(
    LON, LAT, SSH, nperseg=256, noverlap=128, dx_km=None,
    agg='mean', min_segments=1, verbose=True, along_axis="auto",
    fillvalue_candidates=(-32768.0, 1e20, 9.96921e36)
):

    assert SSH.ndim == 2, "SSH doit être 2D"
    SSH = np.array(SSH, dtype=float, copy=True)

    # Sanitize: remplace les fillvalues et non-finies par NaN
    for fv in fillvalue_candidates:
        SSH[np.isclose(SSH, fv, rtol=0, atol=0)] = np.nan
    SSH[~np.isfinite(SSH)] = np.nan

    ny, nx = SSH.shape
    if nperseg <= 1:
        raise ValueError("nperseg doit être > 1.")

    # Estimation dx_km si non fourni
    if dx_km is None:
        dlat = np.diff(LAT, axis=1)
        dlon = np.diff(LON, axis=1)
        lat_c = 0.5 * (LAT[:, :-1] + LAT[:, 1:])
        dy_km = dlat * 111.194
        dx_km_loc = dlon * 111.194 * np.cos(np.deg2rad(lat_c))
        ds_km = np.hypot(dx_km_loc, dy_km)
        dx_km = np.nanmedian(ds_km)
        if not np.isfinite(dx_km) or dx_km <= 0:
            raise ValueError("Impossible d'estimer dx_km.")

    finite_mask = np.isfinite(SSH)

    def longest_run_lengths(mask2d, axis):
        # axis=1: runs le long des colonnes (lignes fixes) ; axis=0: le long des lignes (colonnes fixes)
        arr = mask2d if axis == 1 else mask2d.T
        out = []
        for row in arr:
            if not np.any(row):
                out.append(0); continue
            edges = np.diff(row.astype(np.int8), prepend=0, append=0)
            s = np.where(edges == 1)[0]; e = np.where(edges == -1)[0]
            out.append(int(np.max(e - s)) if s.size else 0)
        return np.asarray(out, dtype=int)

    runs_row = longest_run_lengths(finite_mask, axis=1)  # avancer le long de x (lignes)
    runs_col = longest_run_lengths(finite_mask, axis=0)  # avancer le long de y (colonnes)
    max_run_row = runs_row.max() if runs_row.size else 0
    max_run_col = runs_col.max() if runs_col.size else 0

    def nearest_pow2_leq(n):
        if n <= 1:
            return 1
        return 1 << int(np.floor(np.log2(n)))

    # --- Choix d'axe et calcul de runmax (UTILISÉ ensuite) ---
    if along_axis == "auto":
        if max_run_row >= max_run_col:
            use_axis = 1
            runmax = max_run_row
        else:
            use_axis = 0
            runmax = max_run_col
    elif along_axis in (0, 1):
        use_axis = along_axis
        runmax = max_run_col if use_axis == 0 else max_run_row
    else:
        raise ValueError("along_axis doit valoir 'auto', 0 ou 1.")

    # Downshift nperseg en fonction de runmax
    nps_eff = min(nperseg, runmax)
    nps_eff = nearest_pow2_leq(nps_eff)

    # Si trop petit, tente l'autre axe une fois
    if nps_eff < 16:
        alt_axis = 0 if use_axis == 1 else 1
        alt_runmax = max_run_col if alt_axis == 0 else max_run_row
        alt_nps = nearest_pow2_leq(min(nperseg, alt_runmax))
        if alt_nps >= 16:
            use_axis, runmax, nps_eff = alt_axis, alt_runmax, alt_nps
        else:
            if verbose:
                print(f"[welch] Aucun run suffisant. max_run_row={max_run_row}, max_run_col={max_run_col}")
            return np.array([]), np.array([])

    # Recalcule noverlap/step compatibles avec nps_eff
    noverlap = min(noverlap, nps_eff - 1)
    step = max(1, nps_eff - noverlap)

    # Grille et fenêtre pour nps_eff
    k_cpkm = rfftfreq(nps_eff, d=dx_km)
    w = np.hanning(nps_eff); w2_sum = np.sum(w**2)

    def windows_from_segments(vec):
        all_psd = []; n_added = 0
        ok = np.isfinite(vec)
        if ok.sum() < nps_eff:
            return 0, all_psd
        edges = np.diff(ok.astype(np.int8), prepend=0, append=0)
        starts = np.where(edges == 1)[0]; ends = np.where(edges == -1)[0]
        for a, b in zip(starts, ends):
            n = b - a
            if n < nps_eff:
                continue
            seg = vec[a:b]
            for i0 in range(0, n - nps_eff + 1, step):
                win = seg[i0:i0+nps_eff]
                if not np.all(np.isfinite(win)):
                    continue
                win = win - np.mean(win)
                X = rfft(win * w)
                psd = (np.abs(X)**2) / (w2_sum * dx_km)
                all_psd.append(psd); n_added += 1
        return n_added, all_psd

    # Balayage selon l'axe retenu
    all_psd = []; total_windows = 0
    if use_axis == 1:
        for iy in range(ny):
            n, psds = windows_from_segments(SSH[iy, :])
            total_windows += n; all_psd.extend(psds)
    else:
        for ix in range(nx):
            n, psds = windows_from_segments(SSH[:, ix])
            total_windows += n; all_psd.extend(psds)

    if verbose:
        axname = "lignes (axis=1)" if use_axis == 1 else "colonnes (axis=0)"
        med_row = int(np.median(runs_row)) if runs_row.size else 0
        med_col = int(np.median(runs_col)) if runs_col.size else 0
        #print(f"[welch] dx_km≈{dx_km:.5g} km, axe choisi: {axname}, runmax={int(runmax)}")
        #print(f"[welch] run max (axis=1): median={med_row}, max={int(max_run_row)}")
        #print(f"[welch] run max (axis=0): median={med_col}, max={int(max_run_col)}")
        #print(f"[welch] nperseg effectif: {nps_eff} (step={step}, noverlap={noverlap})")
        #print(f"[welch] fenêtres retenues: {total_windows}")

    if total_windows < min_segments or len(all_psd) == 0:
        if verbose:
            print(f"[welch] Trop peu de fenêtres valides (min={min_segments}).")
        return np.array([]), np.array([])

    S = np.stack(all_psd, axis=0)
    E = np.nanmedian(S, axis=0) if agg == 'median' else np.nanmean(S, axis=0)
    return k_cpkm, E





# ---------- Strain / vorticity on bins ----------
def strain_vort_binned(LON, LAT, U, V, lon_edges, lat_edges, res=0.02, min_count=3):
    # distances on native grid
    dx_right, dy_down = grid_dx_dy(LON, LAT)
    dudX, dudY = central_gradient(U, dx_right, dy_down)
    dvdX, dvdY = central_gradient(V, dx_right, dy_down)
    # project to East/North
    dudx, dudy = projection_gradients(dudX, dudY, LON, LAT)
    dvdx, dvdy = projection_gradients(dvdX, dvdY, LON, LAT)
    # coriolis
    f = 2 * OMEGA * np.sin(np.deg2rad(LAT))
    with np.errstate(invalid="ignore", divide="ignore"):
        vort_norm = (dvdx - dudy) / f
    strain = Strain(dudx, dudy, dvdx, dvdy)

    # flatten valid Med
    lonf = LON.ravel(); latf = LAT.ravel()
    Sf   = strain.ravel()
    Zf   = vort_norm.ravel()


    # strain
    ny = lat_edges.size - 1; nx = lon_edges.size - 1
    sumS = np.zeros((ny, nx)); cntS = np.zeros((ny, nx), dtype=int)
    sumZ = np.zeros((ny, nx)); cntZ = np.zeros((ny, nx), dtype=int)
    accumulate_bins(lonf, latf, Sf, lon_edges, lat_edges, sumS, cntS)
    accumulate_bins(lonf, latf, Zf, lon_edges, lat_edges, sumZ, cntZ)

    with np.errstate(invalid="ignore", divide="ignore"):
        Smean = sumS / cntS
        Zmean = sumZ / cntZ
    Smean[cntS < min_count] = np.nan
    Zmean[cntZ < min_count] = np.nan
    return Smean, Zmean

def _raise_gridlines(gl, z=10):
    for coll in (getattr(gl, "xlines", None), getattr(gl, "ylines", None)):
        if coll is None:
            continue
        # cas LineCollection (unique)
        if hasattr(coll, "set_zorder"):
            coll.set_zorder(z)
        else:
            # cas liste d'artistes
            try:
                for artist in coll:
                    artist.set_zorder(z)
            except TypeError:
                pass

    # Monter les labels si présents
    for labs in (getattr(gl, "xlabel_artists", []), getattr(gl, "ylabel_artists", [])):
        try:
            for lab in labs:
                lab.set_zorder(z + 0.1)
        except TypeError:
            pass

    # Au cas où Cartopy respecte gl.zorder :
    try:
        gl.zorder = z
    except Exception:
        pass
# ---------- Plot helpers ----------
def plot_map(lon_c, lat_c, field, title, savefig=None, show=False, cmap="RdBu_r", vmin=None, vmax=None, label=None):
    import matplotlib.ticker as mticker
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    proj_map = ccrs.Mercator()
    proj_data = ccrs.PlateCarree()  # données en lon/lat

    fig = plt.figure(figsize=(10, 5), layout="constrained")
    ax = plt.axes(projection=proj_map)
    ax.set_extent([-6, 36, 30, 46], crs=proj_data) #full med

    land = cfeature.NaturalEarthFeature("physical","land","50m", edgecolor="black", facecolor="lightgray")
    ax.add_feature(land); ax.coastlines(resolution="50m", color="black", linewidth=0.6)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(np.arange(-6, 37, 6)) #full med
    gl.ylocator = mticker.FixedLocator(np.arange(30, 47, 2)) #full med

    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    tick_fontsize = 12
    gl.xlabel_style = {'size': tick_fontsize}
    gl.ylabel_style = {'size': tick_fontsize}

    _raise_gridlines(gl, z=10)

    dlon = np.diff(lon_c).mean() if lon_c.size>1 else 0.02
    dlat = np.diff(lat_c).mean() if lat_c.size>1 else 0.02
    lon_edges = np.concatenate(([lon_c[0]-0.5*dlon], 0.5*(lon_c[:-1]+lon_c[1:]), [lon_c[-1]+0.5*dlon]))
    lat_edges = np.concatenate(([lat_c[0]-0.5*dlat], 0.5*(lat_c[:-1]+lat_c[1:]), [lat_c[-1]+0.5*dlat]))
    LON, LAT = np.meshgrid(lon_edges, lat_edges)
    pcm = ax.pcolormesh(LON, LAT, field, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    cbar = fig.colorbar(pcm, ax=ax, shrink=0.6,extend='both')
    if label: 
        cbar.set_label(label,fontsize=tick_fontsize)
        cbar.ax.tick_params(labelsize=tick_fontsize)
    ax.set_title(title)
    if savefig:
        Path(savefig).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savefig, dpi=300)
    if show: plt.show()
    else: plt.close(fig)

# ---------- Orchestrator ----------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="SSH spectrum + PDFs of strain and normalized vorticity per L3 cycle (Mediterranean).")
    ap.add_argument("--l3_root", required=True, help="Folder with cycle_XXX subfolders (from run_cycles_swot.py)")
    ap.add_argument("--res", type=float, default=0.025, help="bin size in degrees for binned fields (default 0.02 ≈ 2 km)")
    ap.add_argument("--min-count", type=int, default=1, help="min obs per bin to keep bin")
    ap.add_argument("--nperseg", type=int, default=512, help="Welch segment length for SSH spectra")
    ap.add_argument("--noverlap", type=int, default=256, help="Welch overlap for SSH spectra")
    ap.add_argument("--area-weight", action="store_true", help="Area-weight PDFs by cos(lat) (recommended)")
    ap.add_argument("--savefigs", action="store_true", help="Save per-cycle maps of strain and vortnorm")
    ap.add_argument("--outdir", default="derived_metrics", help="Output folder for CSVs and optional figures")
    args = ap.parse_args()

    l3_root = Path(args.l3_root)
    outdir  = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # CSV writers (long format)
    spectra_csv   = (outdir / "ssh_spectra.csv").open("w", newline="")
    strain_csv    = (outdir / "pdf_strain.csv").open("w", newline="")
    vortnorm_csv  = (outdir / "pdf_vort_norm.csv").open("w", newline="")
    w_spec = csv.DictWriter(spectra_csv, fieldnames=["cycle","k_cpkm","E_ssh"])
    w_str  = csv.DictWriter(strain_csv,  fieldnames=["cycle","bin_center","pdf"])
    w_vor  = csv.DictWriter(vortnorm_csv,fieldnames=["cycle","bin_center","pdf"])
    w_spec.writeheader(); w_str.writeheader(); w_vor.writeheader()

    # grid for binning maps/PDFs
    lon_edges, lat_edges, lon_c, lat_c = make_grid(res=args.res)
    ny, nx = lat_c.size, lon_c.size
    weights_line = np.cos(np.deg2rad(lat_c))[:, None]  # for area-weighted PDFs

    # discover cycles
    cycles = sorted([int(p.name.split("_")[-1]) for p in l3_root.glob("cycle_*") if p.is_dir()])

    for c in cycles:
        cdir = l3_root / f"cycle_{c:03d}"
        files = sorted(cdir.glob("SWOT_L3_LR_SSH_Expert_*.nc"))
        if not files:
            continue

        print(f"\n=== Cycle {c} ===")

        # --- accumulators for PDFs/maps ---
        sumS = np.zeros((ny, nx)); cntS = np.zeros((ny, nx), dtype=int)
        sumZ = np.zeros((ny, nx)); cntZ = np.zeros((ny, nx), dtype=int)

        # --- accumulateurs pour les PDF instantanées (sur tous les fichiers du cycle) ---
        nbins_pdf = 200
    
        # bornes à ajuster si besoin
        edges_S = np.linspace(0.0, 3.0e-4, nbins_pdf + 1)   # strain ~ O(10⁻⁴ s⁻¹)
        edges_Z = np.linspace(-3.0, 3.0, nbins_pdf + 1)     # ζ/f typiquement |ζ/f| < 3
    
        histS = np.zeros(nbins_pdf, dtype=float)
        histZ = np.zeros(nbins_pdf, dtype=float)

        # --- spectra accumulators ---
        # --- spectra accumulators ---
        k_ref = None
        Ek_sum = None
        n_spec_files = 0  # <-- nouveau
        
        for f in tqdm(files, desc=f"cycle {c:03d}"):
            out = read_vars_from_nc(f)
            if out is None:
                print(f"[WARN] {Path(f).name}: lecture impossible (vars manquantes ou masque vide)")
                continue
            LON, LAT, SSH, U, V = out
        
            # 1) spectra (row-wise Welch)
            k_cpkm, E = welch_rowwise_psd(LON, LAT, SSH, nperseg=args.nperseg, noverlap=args.noverlap)
            if k_cpkm.size and np.all(np.isfinite(E)):
                if k_ref is None:
                    k_ref  = k_cpkm
                    Ek_sum = np.zeros_like(k_ref)
            
                kmin = max(k_ref[0],  k_cpkm[0])
                kmax = min(k_ref[-1], k_cpkm[-1])
                mref = (k_ref >= kmin) & (k_ref <= kmax)
                if mref.sum() >= 8:  # recouvrement minimal
                    Ek_sum[mref] += np.interp(k_ref[mref], k_cpkm, E)
                    n_spec_files += 1
                else:
                    print(f"[DBG] {Path(f).name}: recouvrement k trop faible (ignoré)")
            else:
                print(f"[WARN] {Path(f).name}: spectre vide ou non fini (ignoré)")
                    
            # 2) S, vort/f binned
            Smean, Zmean = strain_vort_binned(LON, LAT, U, V, lon_edges, lat_edges, res=args.res, min_count=args.min_count)
            #print(f"[DBG2] {Path(f).name}: bins Smean={np.isfinite(Smean).sum()}, Zmean={np.isfinite(Zmean).sum()}")
            # accumulate mean-by-bin (sum/count)
            mS = np.isfinite(Smean)
            mZ = np.isfinite(Zmean)
            sumS[mS] += Smean[mS]; cntS[mS] += 1
            sumZ[mZ] += Zmean[mZ]; cntZ[mZ] += 1

        # --- PDF instantanée : on ajoute directement tous les bins de CE fichier ---
            # strain
            if np.any(mS):
                valsS = Smean[mS]
                if args.area_weight:
                    iy, ix = np.where(mS)
                    wS = weights_line[iy, 0]          # cos(lat) pour chaque bin
                    hS, _ = np.histogram(valsS, bins=edges_S, weights=wS)
                else:
                    hS, _ = np.histogram(valsS, bins=edges_S)
                histS += hS
        
            # vorticité normalisée
            if np.any(mZ):
                valsZ = Zmean[mZ]
                if args.area_weight:
                    iy, ix = np.where(mZ)
                    wZ = weights_line[iy, 0]
                    hZ, _ = np.histogram(valsZ, bins=edges_Z, weights=wZ)
                else:
                    hZ, _ = np.histogram(valsZ, bins=edges_Z)
                histZ += hZ

        if k_ref is not None and n_spec_files > 0:
            Ek_mean = Ek_sum / n_spec_files
            for ki, Ei in zip(k_ref, Ek_mean):
                w_spec.writerow({"cycle": c, "k_cpkm": f"{ki:.6f}", "E_ssh": f"{Ei:.6e}"})
        else:
            print(f"[INFO] Cycle {c:03d}: aucun segment spectral retenu")
    
        with np.errstate(invalid="ignore", divide="ignore"):
            Smap = sumS / cntS
            Zmap = sumZ / cntZ
        Smap[cntS < 1] = np.nan
        Zmap[cntZ < 1] = np.nan
    
        # strain
        dS = edges_S[1] - edges_S[0]
        if histS.sum() > 0:
            pdfS = histS / (histS.sum() * dS)
            centers_S = 0.5 * (edges_S[:-1] + edges_S[1:])
            for xi, pi in zip(centers_S, pdfS):
                w_str.writerow({"cycle": c, "bin_center": f"{xi:.6e}", "pdf": f"{pi:.6e}"})
        else:
            print(f"[INFO] Cycle {c:03d}: aucun échantillon de strain pour la PDF")
    
        # vorticité normalisée ζ/f
        dZ = edges_Z[1] - edges_Z[0]
        if histZ.sum() > 0:
            pdfZ = histZ / (histZ.sum() * dZ)
            centers_Z = 0.5 * (edges_Z[:-1] + edges_Z[1:])
            for xi, pi in zip(centers_Z, pdfZ):
                w_vor.writerow({"cycle": c, "bin_center": f"{xi:.6e}", "pdf": f"{pi:.6e}"})
        else:
            print(f"[INFO] Cycle {c:03d}: aucun échantillon ζ/f pour la PDF")

        # optional figures
        if args.savefigs:
            plot_map(lon_c, lat_c, Smap, f"Cycle {c:03d} — mean strain",
                     savefig=str(outdir / f"map_strain_c{c:03d}.png"),
                     show=False, cmap="viridis", vmin=0, vmax=1.5e-4, label="s⁻¹")
            plot_map(lon_c, lat_c, Zmap, f"Cycle {c:03d} — mean ζ/f",
                     savefig=str(outdir / f"map_vortnorm_c{c:03d}.png"),
                     show=False, cmap="RdBu_r", vmin=-1, vmax=1, label="ζ/f")

    spectra_csv.close(); strain_csv.close(); vortnorm_csv.close()
    print(f"\nSaved CSVs in: {outdir}")

if __name__ == "__main__":
    main()
