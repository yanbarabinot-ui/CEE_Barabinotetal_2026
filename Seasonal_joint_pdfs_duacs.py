from __future__ import annotations

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DUACS-on-SWOT-like sampling: seasonal joint-PDF contours (S/f, zeta/f) per SWOT L3 cycle (Mediterranean)
"""

import re
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from datetime import datetime, date

import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable

import shapely
from shapely.geometry import box
from shapely import contains_xy

import copernicusmarine as cm

# ---------- Constants ----------
EARTH_RADIUS_M = 6371000.0
OMEGA = 7.292115e-5  # rad/s

# ---------- Mediterranean ROI ----------
def build_med_mask():
    med_bbox = box(-6, 30, 36, 46)
    black_sea = box(27, 41, 42, 47)
    bay_biscay = box(-6, 43, -1, 46)
    roi = shapely.difference(med_bbox, shapely.unary_union([black_sea, bay_biscay]))
    return shapely.buffer(roi, 0)

ROI_POLY = build_med_mask()

def to_m180_180(lon):
    return ((lon + 180) % 360) - 180

# ---------- L3 date parsing (for matching DUACS daily) ----------
FN_L3_DATE = re.compile(r"_(\d{8})T\d{6}_")  # ..._YYYYMMDDTHHMMSS_

def l3_file_date(nc_path: Path) -> Optional[date]:
    m = FN_L3_DATE.search(nc_path.name)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d").date()

def l3_cycle_date_range(l3_dir: Path) -> Tuple[date, date]:
    days = []
    for p in l3_dir.glob("SWOT_L3_LR_SSH_Expert_*.nc"):
        d = l3_file_date(p)
        if d is not None:
            days.append(d)
    if not days:
        raise RuntimeError(f"No dates found in {l3_dir}")
    return min(days), max(days)

# ---------- Grid & binning ----------
def make_grid(lon_min=-6, lon_max=36, lat_min=30, lat_max=46, res=0.02):
    lon_edges = np.arange(lon_min, lon_max + 1e-12, res)
    lat_edges = np.arange(lat_min, lat_max + 1e-12, res)
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
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

# ---------- Helper functions ----------
def row_angle_from_finite_ends(i, LON, LAT, return_degrees=False):
    lon_row = LON[i, :]
    lat_row = LAT[i, :]
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
    if return_degrees:
        ang = np.rad2deg(ang)
    return ang

def grid_dx_dy(lon: np.ndarray, lat: np.ndarray):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lam = np.deg2rad(lon)
    phi = np.deg2rad(lat)
    cosphi = np.cos(phi)

    dlam_r = np.empty_like(lam)
    dphi_r = np.empty_like(phi)
    dlam_r[:, :-1] = lam[:, 1:] - lam[:, :-1]
    dphi_r[:, :-1] = phi[:, 1:] - phi[:, :-1]
    dlam_r[:, -1]  = dlam_r[:, -2]
    dphi_r[:, -1]  = dphi_r[:, -2]
    dx_right = EARTH_RADIUS_M * np.sqrt(dphi_r**2 + (cosphi * dlam_r)**2)

    dlam_d = np.empty_like(lam)
    dphi_d = np.empty_like(phi)
    dlam_d[:-1, :] = lam[1:, :] - lam[:-1, :]
    dphi_d[:-1, :] = phi[1:, :] - phi[:-1, :]
    dlam_d[-1, :]  = dlam_d[-2, :]
    dphi_d[-1, :]  = dphi_d[-2, :]
    dy_down = EARTH_RADIUS_M * np.sqrt(dphi_d**2 + (cosphi * dlam_d)**2)

    return dx_right, dy_down

def central_gradient(a: np.ndarray, dx_right: np.ndarray, dy_down: np.ndarray):
    a = np.asarray(a, dtype=float)
    dax = np.full_like(a, np.nan)
    day = np.full_like(a, np.nan)

    dx_left = np.empty_like(dx_right)
    dx_left[:, 1:] = dx_right[:, :-1]
    dx_left[:, 0]  = dx_right[:, 0]
    dx_sum = dx_left + dx_right

    A = np.isfinite(a)

    m_c = A[:, 2:] & A[:, :-2] & np.isfinite(dx_sum[:, 1:-1]) & (dx_sum[:, 1:-1] > 0)
    dax[:, 1:-1][m_c] = (a[:, 2:][m_c] - a[:, :-2][m_c]) / dx_sum[:, 1:-1][m_c]

    m_f = A[:, 1] & A[:, 0] & np.isfinite(dx_right[:, 0]) & (dx_right[:, 0] > 0)
    dax[:, 0][m_f] = (a[:, 1][m_f] - a[:, 0][m_f]) / dx_right[:, 0][m_f]

    m_b = A[:, -1] & A[:, -2] & np.isfinite(dx_left[:, -1]) & (dx_left[:, -1] > 0)
    dax[:, -1][m_b] = (a[:, -1][m_b] - a[:, -2][m_b]) / dx_left[:, -1][m_b]

    dy_up = np.empty_like(dy_down)
    dy_up[1:, :] = dy_down[:-1, :]
    dy_up[0, :]  = dy_down[0, :]
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
        ang = row_angle_from_finite_ends(i, LON, LAT)
        if not np.isfinite(ang):
            continue
        c, s = np.cos(ang), np.sin(ang)
        dudx[i, :] = c * dudX[i, :] - s * dudY[i, :]
        dudy[i, :] = s * dudX[i, :] + c * dudY[i, :]
    return dudx, dudy

def Strain(dudx, dudy, dvdx, dvdy):
    s_n = dudx - dvdy
    s_s = dudy + dvdx
    return np.sqrt(s_n**2 + s_s**2)

def strain_vort_binned(LON, LAT, U, V, lon_edges, lat_edges, min_count=3):
    dx_right, dy_down = grid_dx_dy(LON, LAT)
    dudX, dudY = central_gradient(U, dx_right, dy_down)
    dvdX, dvdY = central_gradient(V, dx_right, dy_down)

    dudx, dudy = projection_gradients(dudX, dudY, LON, LAT)
    dvdx, dvdy = projection_gradients(dvdX, dvdY, LON, LAT)

    f = 2 * OMEGA * np.sin(np.deg2rad(LAT))
    with np.errstate(invalid="ignore", divide="ignore"):
        vort_norm = (dvdx - dudy) / f
    strain = Strain(dudx, dudy, dvdx, dvdy)

    lonf = LON.ravel()
    latf = LAT.ravel()
    Sf = strain.ravel()
    Zf = vort_norm.ravel()

    ny = lat_edges.size - 1
    nx = lon_edges.size - 1
    sumS = np.zeros((ny, nx))
    cntS = np.zeros((ny, nx), dtype=int)
    sumZ = np.zeros((ny, nx))
    cntZ = np.zeros((ny, nx), dtype=int)

    accumulate_bins(lonf, latf, Sf, lon_edges, lat_edges, sumS, cntS)
    accumulate_bins(lonf, latf, Zf, lon_edges, lat_edges, sumZ, cntZ)

    with np.errstate(invalid="ignore", divide="ignore"):
        Smean = sumS / cntS
        Zmean = sumZ / cntZ

    Smean[cntS < min_count] = np.nan
    Zmean[cntZ < min_count] = np.nan
    return Smean, Zmean

# =============================================================================
# DUACS + interpolation onto SWOT swath
# =============================================================================

DUACS_DAILY_DATASET_ID = "cmems_obs-sl_eur_phy-ssh_my_allsat-l4-duacs-0.0625deg_P1D"
FN_DUACS_DATE = re.compile(r"(\d{8})")

def month_range(d0: date, d1: date) -> List[Tuple[int, int]]:
    out = []
    y, m = d0.year, d0.month
    while True:
        out.append((y, m))
        if y == d1.year and m == d1.month:
            break
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out

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
            skip_existing=True,
        )
    except Exception as e:
        print(f"[WARN] cm.get failed for {year}-{month:02d}: {e}")

    for nc in rawdir.rglob("*.nc"):
        target = out_flat / nc.name
        if not target.exists():
            try:
                nc.replace(target)
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
                try:
                    p.rmdir()
                except OSError:
                    pass
        rawdir.rmdir()
    except Exception:
        pass

def _read_day_from_duacs(ds: Dataset, nc_path: Path) -> Optional[date]:
    tvar = next((nm for nm in ("time", "day", "t") if nm in ds.variables), None)
    if tvar:
        t = np.array(ds.variables[tvar][:]).ravel()
        if t.size:
            try:
                units = getattr(ds.variables[tvar], "units", "")
                calendar = getattr(ds.variables[tvar], "calendar", "standard")
                import cftime
                dts = cftime.num2date(t, units=units, calendar=calendar)
                dt0 = dts[0] if np.ndim(dts) else dts
                return date(int(dt0.year), int(dt0.month), int(dt0.day))
            except Exception:
                pass

    m = FN_DUACS_DATE.search(nc_path.name)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d").date()
        except Exception:
            return None
    return None

def read_duacs_daily_fields(
    nc_path: Path,
    ssh_candidates=("sla", "ssha"),
    u_candidates=("ugosa"),
    v_candidates=("vgosa"),
):
    """
    Returns lon1d, lat1d, SSH2D, U2D, V2D, day
    """
    try:
        with Dataset(nc_path, "r") as ds:
            ds.set_auto_maskandscale(True)

            vlon = "longitude" if "longitude" in ds.variables else ("lon" if "lon" in ds.variables else None)
            vlat = "latitude"  if "latitude"  in ds.variables else ("lat" if "lat" in ds.variables else None)
            if vlon is None or vlat is None:
                return None

            lon = to_m180_180(np.array(ds.variables[vlon][:], dtype="float64"))
            lat = np.array(ds.variables[vlat][:], dtype="float64")

            ssh_name = next((nm for nm in ssh_candidates if nm in ds.variables), None)
            u_name = next((nm for nm in u_candidates if nm in ds.variables), None)
            v_name = next((nm for nm in v_candidates if nm in ds.variables), None)
            if ssh_name is None or u_name is None or v_name is None:
                return None

            SSHv = ds.variables[ssh_name]
            Uv = ds.variables[u_name]
            Vv = ds.variables[v_name]

            def read_nan(var):
                arr = var[:]
                return (arr.filled(np.nan) if np.ma.isMaskedArray(arr) else np.array(arr)).astype("float64")

            ssh_raw = SSHv[0, ...] if SSHv.ndim == 3 else SSHv[:]
            u_raw = Uv[0, ...] if Uv.ndim == 3 else Uv[:]
            v_raw = Vv[0, ...] if Vv.ndim == 3 else Vv[:]

            ssh = read_nan(ssh_raw)
            u = read_nan(u_raw)
            v = read_nan(v_raw)

            def to_mps(arr, var):
                units = (getattr(var, "units", "") or "").lower().replace(" ", "")
                if "cm/s" in units or "cms-1" in units or "cms^-1" in units:
                    return arr / 100.0
                return arr

            u = to_mps(u, Uv)
            v = to_mps(v, Vv)

            if lon.ndim == 2:
                lon = lon[0, :]
            if lat.ndim == 2:
                lat = lat[:, 0]

            if lon.size < 2 or lat.size < 2:
                return None
            if not (np.all(np.diff(lon) > 0) and np.all(np.diff(lat) > 0)):
                return None

            day = _read_day_from_duacs(ds, nc_path)
            return lon, lat, ssh, u, v, day
    except Exception:
        return None

def bilinear_interp_regular(lon1d, lat1d, field2d, lonp, latp):
    lon1d = np.asarray(lon1d)
    lat1d = np.asarray(lat1d)
    F = np.asarray(field2d)

    ix = np.searchsorted(lon1d, lonp) - 1
    iy = np.searchsorted(lat1d, latp) - 1

    ok = (ix >= 0) & (ix < lon1d.size - 1) & (iy >= 0) & (iy < lat1d.size - 1)
    out = np.full_like(lonp, np.nan, dtype=float)
    if not np.any(ok):
        return out

    x0 = lon1d[ix[ok]]
    x1 = lon1d[ix[ok] + 1]
    y0 = lat1d[iy[ok]]
    y1 = lat1d[iy[ok] + 1]

    tx = (lonp[ok] - x0) / (x1 - x0)
    ty = (latp[ok] - y0) / (y1 - y0)

    f00 = F[iy[ok], ix[ok]]
    f10 = F[iy[ok], ix[ok] + 1]
    f01 = F[iy[ok] + 1, ix[ok]]
    f11 = F[iy[ok] + 1, ix[ok] + 1]

    good = np.isfinite(f00) & np.isfinite(f10) & np.isfinite(f01) & np.isfinite(f11)
    if np.any(good):
        txg = tx[good]
        tyg = ty[good]
        v = (1 - txg) * (1 - tyg) * f00[good] + txg * (1 - tyg) * f10[good] + (1 - txg) * tyg * f01[good] + txg * tyg * f11[good]
        tmp = out[ok]
        tmp[good] = v
        out[ok] = tmp

    return out

def read_swot_geom_and_mask(
    nc_path: Path,
    swot_ssh_for_mask_candidates=("ssha_filtered", "ssha", "ssh", "sea_surface_height"),
):

    with Dataset(nc_path, "r") as ds:
        vlon = "longitude" if "longitude" in ds.variables else ("lon" if "lon" in ds.variables else None)
        vlat = "latitude"  if "latitude"  in ds.variables else ("lat" if "lat" in ds.variables else None)
        if vlon is None or vlat is None:
            return None

        lon = to_m180_180(np.array(ds.variables[vlon][:], dtype="float64"))
        lat = np.array(ds.variables[vlat][:], dtype="float64")

        if lon.ndim == 1 and lat.ndim == 1:
            LON, LAT = np.meshgrid(lon, lat)
        else:
            LON, LAT = lon, lat

        mask = np.isfinite(LON) & np.isfinite(LAT)

        if "quality_flag" in ds.variables:
            q = np.array(ds.variables["quality_flag"][:])
            mask &= (q == 0)

        for vv in swot_ssh_for_mask_candidates:
            if vv in ds.variables:
                arr = np.array(ds.variables[vv][:], dtype="float64")
                mask &= np.isfinite(arr)
                break

        lonf = LON.ravel()
        latf = LAT.ravel()
        finite_xy = np.isfinite(lonf) & np.isfinite(latf)
        inside_flat = np.zeros_like(finite_xy, dtype=bool)
        inside_flat[finite_xy] = contains_xy(ROI_POLY, lonf[finite_xy], latf[finite_xy])
        inside = inside_flat.reshape(LON.shape)
        mask &= inside

        return LON, LAT, mask

def duacs_on_swot(LON, LAT, mask, du_lon, du_lat, du_ssh, du_u, du_v):
    SSHi = bilinear_interp_regular(du_lon, du_lat, du_ssh, LON, LAT)
    Ui = bilinear_interp_regular(du_lon, du_lat, du_u, LON, LAT)
    Vi = bilinear_interp_regular(du_lon, du_lat, du_v, LON, LAT)

    SSHi[~mask] = np.nan
    Ui[~mask] = np.nan
    Vi[~mask] = np.nan
    return SSHi, Ui, Vi


def load_cycle_month_map(cycle_info_csv: Path) -> Dict[int, int]:
    df = pd.read_csv(cycle_info_csv, parse_dates=["date_min", "date_max"])
    df["cycle"] = df["cycle"].astype(int)
    df["date_median"] = df["date_min"] + (df["date_max"] - df["date_min"]) / 2.0
    df["month"] = df["date_median"].dt.month.astype(int)
    return dict(zip(df["cycle"].values, df["month"].values))

def hdr_threshold(pdf2d: np.ndarray, dZ: float, dS: float, mass: float = 0.5) -> float:
    p = np.asarray(pdf2d, dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return np.nan
    p_sorted = np.sort(p)[::-1]
    cum = np.cumsum(p_sorted) * dZ * dS
    idx = np.searchsorted(cum, mass, side="left")
    if idx >= p_sorted.size:
        return p_sorted[-1]
    return p_sorted[idx]

# =============================================================================
# main
# =============================================================================

def main():
    import argparse
    ap = argparse.ArgumentParser(description="DUACS-on-SWOT-like: seasonal joint-PDF HDR contours (Mediterranean).")
    ap.add_argument("--l3_root", required=True, help="Folder with cycle_XXX subfolders (SWOT L3; used for geometry+mask)")
    ap.add_argument("--duacs_cache", default="duacs_l4_cache", help="Cache folder for DUACS daily netCDF")
    ap.add_argument("--res", type=float, default=0.025, help="bin size in degrees for binned fields (default 0.025)")
    ap.add_argument("--min-count", type=int, default=1, help="min obs per bin to keep bin")
    ap.add_argument("--area-weight", action="store_true", help="Area-weight joint PDF by cos(lat) (recommended)")

    ap.add_argument("--cycle-info", default=None,
                    help="CSV with cycle/date_min/date_max used for seasonal coloring (optional)")
    ap.add_argument("--hdr-mass", type=float, default=0.5,
                    help="Probability mass enclosed by HDR contour (e.g. 0.5 for 50%%)")
    ap.add_argument("--nbins-strain", type=int, default=200)
    ap.add_argument("--nbins-vort", type=int, default=200)
    ap.add_argument("--strain-max", type=float, default=2.0, help="Max of S/f axis")
    ap.add_argument("--vort-max", type=float, default=2.0, help="Max of zeta/f axis")
    ap.add_argument("--jointpdf-outfile", required=True,
                    help="Save seasonal joint-PDF contour overlay figure to this path (png/pdf)")
    args = ap.parse_args()

    l3_root = Path(args.l3_root)
    duacs_cache = Path(args.duacs_cache)

    lon_edges, lat_edges, lon_c, lat_c = make_grid(res=args.res)
    weights_line = np.cos(np.deg2rad(lat_c))[:, None]

    cycles = sorted([int(p.name.split("_")[-1]) for p in l3_root.glob("cycle_*") if p.is_dir()])

    cycle_month = load_cycle_month_map(Path(args.cycle_info)) if args.cycle_info else {}

    def cycle_month_value(cyc: int) -> int:
        m = cycle_month.get(int(cyc), None)
        if m is None:
            return 1
        return int(m)
    
    nbS = args.nbins_strain
    nbZ = args.nbins_vort
    edges_Sn = np.linspace(0.0, args.strain_max, nbS + 1)
    edges_Zj = np.linspace(-args.vort_max, args.vort_max, nbZ + 1)
    dZj = edges_Zj[1] - edges_Zj[0]
    dSn = edges_Sn[1] - edges_Sn[0]

    Z_centers = 0.5 * (edges_Zj[:-1] + edges_Zj[1:])
    S_centers = 0.5 * (edges_Sn[:-1] + edges_Sn[1:])
    ZZ, SS = np.meshgrid(Z_centers, S_centers)

    joint_per_cycle = []  # (cycle:int, pdf2d:(nbZ,nbS), month:int, thr:float)
    
    for c in cycles:
        cdir = l3_root / f"cycle_{c:03d}"
        files = sorted(cdir.glob("SWOT_L3_LR_SSH_Expert_*.nc"))
        if not files:
            continue

        print(f"\n=== Cycle {c} ===")

        d0, d1 = l3_cycle_date_range(cdir)
        for (yy, mm) in month_range(d0, d1):
            duacs_download_month(yy, mm, duacs_cache, force=False)

        duacs_day_to_file: Dict[date, Path] = {}
        for (yy, mm) in month_range(d0, d1):
            mdir = duacs_cache / f"{yy:04d}" / f"{mm:02d}"
            if not mdir.is_dir():
                continue
            for nc in sorted(mdir.glob("*.nc")):
                try:
                    with Dataset(nc, "r") as ds:
                        ds.set_auto_maskandscale(True)
                        day = _read_day_from_duacs(ds, nc)
                except Exception:
                    day = None
                if day is None or day < d0 or day > d1:
                    continue
                duacs_day_to_file.setdefault(day, nc)

        duacs_cache_day: Dict[date, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        hist2d_joint = np.zeros((nbZ, nbS), dtype=float)  # [zeta_bin, S_bin]

        for f in tqdm(files, desc=f"cycle {c:03d}"):
            day = l3_file_date(f)
            if day is None:
                print(f"[WARN] {f.name}: date introuvable (skip)")
                continue

            duacs_nc = duacs_day_to_file.get(day, None)
            if duacs_nc is None:
                continue

            geom = read_swot_geom_and_mask(f)
            if geom is None:
                continue
            LON, LAT, mask_swot = geom
            if not np.any(mask_swot):
                continue

            if day in duacs_cache_day:
                du_lon, du_lat, du_ssh, du_u, du_v = duacs_cache_day[day]
            else:
                out = read_duacs_daily_fields(duacs_nc)
                if out is None:
                    continue
                du_lon, du_lat, du_ssh, du_u, du_v, _ = out
                duacs_cache_day[day] = (du_lon, du_lat, du_ssh, du_u, du_v)

            _, U, V = duacs_on_swot(LON, LAT, mask_swot, du_lon, du_lat, du_ssh, du_u, du_v)

            Smean, Zmean = strain_vort_binned(
                LON, LAT, U, V,
                lon_edges, lat_edges,
                min_count=args.min_count
            )

            f_line = 2 * OMEGA * np.sin(np.deg2rad(lat_c))
            with np.errstate(invalid="ignore", divide="ignore"):
                Snorm = Smean / f_line[:, None]

            m = np.isfinite(Snorm) & np.isfinite(Zmean)
            if not np.any(m):
                continue

            Svals = Snorm[m]
            Zvals = Zmean[m]  # already zeta/f

            if args.area_weight:
                iy, _ix = np.where(m)
                w = weights_line[iy, 0]
            else:
                w = None

            H2, _, _ = np.histogram2d(
                Zvals, Svals,
                bins=(edges_Zj, edges_Sn),
                weights=w
            )
            hist2d_joint += H2

        total_mass = hist2d_joint.sum() * dZj * dSn
        if total_mass > 0 and np.isfinite(total_mass):
            pdf2d = hist2d_joint / total_mass
            pdf2d[pdf2d == 0] = np.nan
            thr = hdr_threshold(pdf2d, dZ=dZj, dS=dSn, mass=args.hdr_mass)
            if np.isfinite(thr):
                joint_per_cycle.append((int(c), pdf2d, cycle_month_value(c), float(thr)))
        else:
            print(f"[INFO] Cycle {c:03d}: histogramme joint (zeta/f,S/f) vide -> skip contour")

    if len(joint_per_cycle) == 0:
        print("[INFO] Aucun cycle avec joint-PDF valide -> figure non produite.")
        return

    fig, ax = plt.subplots(figsize=(6.2, 5.0), constrained_layout=True)

    base_cmap = plt.get_cmap("twilight")
    months = np.arange(1, 13)
    month_phase = ((months - 2) / 12.0) % 1.0
    month_colors = base_cmap(month_phase)

    month_luminance = {
        m: 0.2126 * month_colors[m-1][0] + 0.7152 * month_colors[m-1][1] + 0.0722 * month_colors[m-1][2]
        for m in range(1, 13)
    }

    cmap = ListedColormap(month_colors, name="twilight_months_feb_edge_aug_center")
    bounds = np.arange(0.5, 13.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)

    joint_per_cycle_sorted = sorted(
        joint_per_cycle,
        key=lambda t: month_luminance[t[2]]
    )

    for (_cyc, pdf2d, month, thr) in joint_per_cycle_sorted:
        color = cmap(norm(month))
        ax.contour(
            ZZ, SS,
            pdf2d.T,
            levels=[thr],
            colors=[color],
            linewidths=1.2,
            alpha=0.9,
        )

    x_line = np.linspace(edges_Zj[0], edges_Zj[-1], 400)
    ax.plot(x_line, x_line, linestyle="--", linewidth=1.0, color="k")
    ax.plot(x_line, -x_line, linestyle="--", linewidth=1.0, color="k")

    ax.set_xlabel(r"$\zeta/f$", fontsize=12)
    ax.set_ylabel(r"$S/f$", fontsize=12)
    ax.set_title("DUACS swotlike", fontsize=12)
    ax.set_xlim(-args.vort_max, args.vort_max)
    ax.set_ylim(0, args.strain_max)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="vertical",
        boundaries=bounds,
        ticks=np.arange(1, 13),
        spacing="proportional",
    )
    cbar.set_label("Month")
    cbar.ax.tick_params(labelsize=12)

    outpath = Path(args.jointpdf_outfile)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Figure seasonal joint-PDF enregistrée dans : {outpath}")

if __name__ == "__main__":
    main()
