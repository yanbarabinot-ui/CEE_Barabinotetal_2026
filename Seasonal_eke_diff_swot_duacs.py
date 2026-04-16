#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 10:28:47 2026

@author: yan
"""

from __future__ import annotations

import re
import csv
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shapely
from shapely.geometry import box
from shapely import contains_xy

# =============================================================================
# parameters
# =============================================================================

L3_ROOT = Path("swot_l3_multi_cycles_v3")
DUACS_CACHE = Path("duacs_l4_cache")
OUTDIR = Path("seasonal_swot_duacs_daily_diff")
OUTDIR.mkdir(parents=True, exist_ok=True)

# reconstruct SWOT grid
RES_FINE = 0.025
LON_MIN, LON_MAX = -6.0, 36.0
LAT_MIN, LAT_MAX = 30.0, 46.0

MIN_COUNT_SWOT_DAY = 1

SAVE_FIG = True
SAVE_NPZ = False
DPI = 300

FN_L3_DATE = re.compile(r"_(\d{8})T\d{6}_")
FN_DUACS_DATE = re.compile(r"(\d{8})")

SEASONS = {
    "winter": {12, 1, 2},
    "summer": {6, 7, 8},
}

def build_med_mask_poly():
    med_bbox   = box(-6, 30, 36, 46)
    black_sea  = box(27, 41, 42, 47)
    bay_biscay = box(-6, 43, -1, 46)
    roi = shapely.difference(med_bbox, shapely.unary_union([black_sea, bay_biscay]))
    return shapely.buffer(roi, 0)

ROI_POLY = build_med_mask_poly()

def to_m180_180(lon):
    return ((lon + 180) % 360) - 180


def make_grid(lon_min=-6, lon_max=36, lat_min=30, lat_max=46, res=0.025):
    lon_edges = np.arange(lon_min, lon_max + 1e-12, res)
    lat_edges = np.arange(lat_min, lat_max + 1e-12, res)
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    return lon_edges, lat_edges, lon_centers, lat_centers

# =============================================================================
# some useful functions
# =============================================================================

def l3_file_date(path: Path) -> Optional[date]:
    m = FN_L3_DATE.search(path.name)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d").date()

def duacs_file_date_from_name(path: Path) -> Optional[date]:
    m = FN_DUACS_DATE.search(path.name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d").date()
    except Exception:
        return None

def parse_cycle_id(path: Path) -> Optional[int]:
    m = re.search(r"cycle_(\d{3})", str(path))
    return int(m.group(1)) if m else None

def cycle_mid_date(days: List[date]) -> date:
    days_sorted = sorted(days)
    return days_sorted[len(days_sorted)//2]

def accumulate_bins(lon, lat, val, lon_edges, lat_edges, sum_grid, cnt_grid):
    ix = np.digitize(lon, lon_edges) - 1
    iy = np.digitize(lat, lat_edges) - 1
    valid = (
        (ix >= 0) & (ix < sum_grid.shape[1]) &
        (iy >= 0) & (iy < sum_grid.shape[0]) &
        np.isfinite(val)
    )
    if not np.any(valid):
        return
    np.add.at(sum_grid, (iy[valid], ix[valid]), val[valid])
    np.add.at(cnt_grid,  (iy[valid], ix[valid]), 1)

def index_l3_by_cycle_and_day(l3_root: Path) -> Dict[int, Dict[date, List[Path]]]:
    out: Dict[int, Dict[date, List[Path]]] = {}
    for f in sorted(l3_root.glob("cycle_*/*.nc")):
        cyc = parse_cycle_id(f)
        d = l3_file_date(f)
        if cyc is None or d is None:
            continue
        out.setdefault(cyc, {}).setdefault(d, []).append(f)
    return out

def index_duacs_by_day(duacs_cache: Path) -> Dict[date, Path]:
    out: Dict[date, Path] = {}
    for f in sorted(duacs_cache.glob("*/*/*.nc")):
        d = duacs_file_date_from_name(f)
        if d is not None:
            out[d] = f
    return out

# =============================================================================
# daily swot
# =============================================================================

def read_swot_daily_points(nc_path: Path):
    with Dataset(nc_path, "r") as ds:
        ds.set_auto_maskandscale(True)

        vlon = "longitude" if "longitude" in ds.variables else ("lon" if "lon" in ds.variables else None)
        vlat = "latitude"  if "latitude"  in ds.variables else ("lat" if "lat" in ds.variables else None)
        if vlon is None or vlat is None:
            raise KeyError(f"lon/lat not found in {nc_path.name}")

        if "ugosa_filtered" not in ds.variables or "vgosa_filtered" not in ds.variables:
            raise KeyError(f"ugosa_filtered/vgosa_filtered not found in {nc_path.name}")

        lon = np.array(np.ma.filled(ds.variables[vlon][:], np.nan), dtype=np.float64)
        lat = np.array(np.ma.filled(ds.variables[vlat][:], np.nan), dtype=np.float64)
        u   = np.array(np.ma.filled(ds.variables["ugosa_filtered"][:], np.nan), dtype=np.float64)
        v   = np.array(np.ma.filled(ds.variables["vgosa_filtered"][:], np.nan), dtype=np.float64)

        lon = to_m180_180(lon)

        # quality_flag 
        valid = np.isfinite(u) & np.isfinite(v)
        if "quality_flag" in ds.variables:
            q = np.array(np.ma.filled(ds.variables["quality_flag"][:], 1))
            valid &= (q == 0)

        valid &= (np.abs(u) < 10.0) & (np.abs(v) < 10.0)

        if lon.ndim == 1 and lat.ndim == 1:
            LON, LAT = np.meshgrid(lon, lat)
        else:
            LON, LAT = lon, lat

        eke = 0.5 * (u*u + v*v)

        lonf = np.asarray(LON).ravel()
        latf = np.asarray(LAT).ravel()
        ekef = np.asarray(eke).ravel()
        valf = np.asarray(valid).ravel()

        finite = np.isfinite(lonf) & np.isfinite(latf) & np.isfinite(ekef) & valf
        inside = np.zeros_like(finite, dtype=bool)
        inside[finite] = contains_xy(ROI_POLY, lonf[finite], latf[finite])

        good = finite & inside
        return lonf[good], latf[good], ekef[good]

def build_swot_daily_map_on_fine_grid(
    l3_files_for_day: List[Path],
    lon_edges: np.ndarray,
    lat_edges: np.ndarray,
    ny: int,
    nx: int,
    min_count: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:

    sum_eke = np.zeros((ny, nx), dtype=np.float64)
    cnt_eke = np.zeros((ny, nx), dtype=np.int64)

    for f in l3_files_for_day:
        try:
            lon, lat, eke = read_swot_daily_points(f)
        except Exception as e:
            print(f"[WARN] SWOT read failed for {f.name}: {e}")
            continue

        if lon.size == 0:
            continue

        accumulate_bins(lon, lat, eke, lon_edges, lat_edges, sum_eke, cnt_eke)

    swot_day = np.full((ny, nx), np.nan, dtype=np.float64)
    valid = cnt_eke >= min_count
    swot_day[valid] = sum_eke[valid] / cnt_eke[valid]

    return swot_day, valid

# =============================================================================
# daily duacs
# =============================================================================

def read_duacs_daily(nc_path: Path):
    with Dataset(nc_path, "r") as ds:
        ds.set_auto_maskandscale(True)

        vlon = "longitude" if "longitude" in ds.variables else ("lon" if "lon" in ds.variables else None)
        vlat = "latitude"  if "latitude"  in ds.variables else ("lat" if "lat" in ds.variables else None)
        if vlon is None or vlat is None:
            raise KeyError(f"lon/lat not found in {nc_path.name}")

        if "ugosa" not in ds.variables or "vgosa" not in ds.variables:
            raise KeyError(f"ugosa/vgosa not found in {nc_path.name}")

        lon = np.array(ds.variables[vlon][:], dtype=np.float64)
        lat = np.array(ds.variables[vlat][:], dtype=np.float64)
        lon = to_m180_180(lon)

        uvar = ds.variables["ugosa"]
        vvar = ds.variables["vgosa"]

        u = np.array(np.ma.filled(uvar[0, ...] if uvar.ndim == 3 else uvar[:], np.nan), dtype=np.float64)
        v = np.array(np.ma.filled(vvar[0, ...] if vvar.ndim == 3 else vvar[:], np.nan), dtype=np.float64)

        units_u = getattr(uvar, "units", "").lower()
        units_v = getattr(vvar, "units", "").lower()
        if ("cm/s" in units_u) or ("cm s-1" in units_u) or ("cm s^-1" in units_u):
            u /= 100.0
        if ("cm/s" in units_v) or ("cm s-1" in units_v) or ("cm s^-1" in units_v):
            v /= 100.0

        eke = 0.5 * (u*u + v*v)

        if lon.ndim == 2 and lat.ndim == 2:
            lon1d = lon[0, :]
            lat1d = lat[:, 0]
        else:
            lon1d = lon
            lat1d = lat

        return lon1d, lat1d, eke

def interp_duacs_to_fine_grid_nearest(
    duacs_field: np.ndarray,
    lon_du: np.ndarray,
    lat_du: np.ndarray,
    lon_fine: np.ndarray,
    lat_fine: np.ndarray,
) -> np.ndarray:

    ix = np.searchsorted(lon_du, lon_fine)
    iy = np.searchsorted(lat_du, lat_fine)

    ix = np.clip(ix, 1, len(lon_du)-1)
    iy = np.clip(iy, 1, len(lat_du)-1)

    ix0 = ix - 1
    iy0 = iy - 1

    choose_x_left = np.abs(lon_fine - lon_du[ix0]) <= np.abs(lon_fine - lon_du[ix])
    choose_y_left = np.abs(lat_fine - lat_du[iy0]) <= np.abs(lat_fine - lat_du[iy])

    ixn = np.where(choose_x_left, ix0, ix)
    iyn = np.where(choose_y_left, iy0, iy)

    out = duacs_field[np.ix_(iyn, ixn)]

    # masque méditerranée sur grille fine
    LONf, LATf = np.meshgrid(lon_fine, lat_fine)
    flat = np.isfinite(LONf.ravel()) & np.isfinite(LATf.ravel())
    inside = np.zeros(LONf.size, dtype=bool)
    inside[flat] = contains_xy(ROI_POLY, LONf.ravel()[flat], LATf.ravel()[flat])
    inside = inside.reshape(LONf.shape)

    out = np.where(inside, out, np.nan)
    return out

def process_cycle_daily_differences(
    cycle_id: int,
    day_to_l3files: Dict[date, List[Path]],
    duacs_by_day: Dict[date, Path],
    lon_edges: np.ndarray,
    lat_edges: np.ndarray,
    lon_fine: np.ndarray,
    lat_fine: np.ndarray,
    ny: int,
    nx: int,
    min_count: int = 1,
):
    diff_sum = np.zeros((ny, nx), dtype=np.float64)
    diff_cnt = np.zeros((ny, nx), dtype=np.int64)
    used_days = []



    for day in sorted(day_to_l3files.keys()):
        if day not in duacs_by_day:
            print(f"[WARN] No DUACS file for {day} (cycle {cycle_id:03d})")
            continue

        swot_day, swot_mask = build_swot_daily_map_on_fine_grid(
            l3_files_for_day=day_to_l3files[day],
            lon_edges=lon_edges,
            lat_edges=lat_edges,
            ny=ny,
            nx=nx,
            min_count=min_count
        )

        if not np.any(swot_mask):
            continue

        try:
            lon_du, lat_du, duacs_eke = read_duacs_daily(duacs_by_day[day])
        except Exception as e:
            print(f"[WARN] DUACS read failed for {day}: {e}")
            continue

        duacs_on_fine = interp_duacs_to_fine_grid_nearest(
            duacs_field=duacs_eke,
            lon_du=lon_du,
            lat_du=lat_du,
            lon_fine=lon_fine,
            lat_fine=lat_fine
        )

        duacs_on_fine = np.where(swot_mask, duacs_on_fine, np.nan)
        swot_day = np.where(swot_mask, swot_day, np.nan)

        diff_day = swot_day - duacs_on_fine
        valid = np.isfinite(diff_day)
        if not np.any(valid):
            continue

        diff_sum[valid] += diff_day[valid]
        diff_cnt[valid] += 1
        used_days.append(day)

    diff_cycle = np.full((ny, nx), np.nan, dtype=np.float64)
    valid = diff_cnt > 0
    diff_cycle[valid] = diff_sum[valid] / diff_cnt[valid]

    return diff_cycle, diff_cnt, used_days

# =============================================================================
# Seasonal mean
# =============================================================================

def assign_cycle_to_season(cycle_days: List[date]) -> Optional[str]:
    if not cycle_days:
        return None
    mid = cycle_mid_date(cycle_days)
    m = mid.month
    if m in SEASONS["winter"]:
        return "winter"
    if m in SEASONS["summer"]:
        return "summer"
    return None

def seasonal_composite_from_cycles(cycle_maps: Dict[int, np.ndarray], cycle_to_season: Dict[int, str]):
    out = {}
    for season in ["winter", "summer"]:
        selected = [cyc for cyc, s in cycle_to_season.items() if s == season and cyc in cycle_maps]
        if not selected:
            out[season] = None
            continue

        arr = np.stack([cycle_maps[cyc] for cyc in selected], axis=0)
        out[season] = np.nanmean(arr, axis=0)
    return out

# =============================================================================
# plots 
# =============================================================================

def compute_edges(x):
    dx = np.diff(x).mean()
    return np.concatenate(([x[0]-0.5*dx], 0.5*(x[:-1]+x[1:]), [x[-1]+0.5*dx]))

def _raise_gridlines(gl, z=10):
    """Rend les gridlines + labels au-dessus, quels que soient les types retournés par Cartopy."""
    # Monter les lignes (x et y)
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

def plot_diff_map(lon, lat, field, title, savefig=None, cmap="RdBu_r"):
    import matplotlib.ticker as mticker
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    proj_map = ccrs.Mercator()
    proj_data = ccrs.PlateCarree()  # données en lon/lat

    fig = plt.figure(figsize=(10, 5), constrained_layout=True)

    ax = plt.axes(projection=proj_map)
    land = cfeature.NaturalEarthFeature("physical", "land", "50m")
    ax.add_feature(land, facecolor="lightgray", edgecolor="black",
                   linewidth=0.5, zorder=3)  # pas de clip_on explicite
    ax.coastlines(resolution="50m", color="black", linewidth=0.6, zorder=4)

    ax.set_extent([-6, 36, 30, 46], crs=ccrs.PlateCarree())
    
    # 3) Graticule avec labels corrects en °E/°N
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(np.arange(-6, 37, 6)) #full med
    gl.ylocator = mticker.FixedLocator(np.arange(30, 47, 2)) #full med
    #gl.xlocator = mticker.FixedLocator(np.arange(-6, -1, 1)) #Alboran
    #gl.ylocator = mticker.FixedLocator(np.arange(35, 37.2, 1)) #Alboran

    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    tick_fontsize = 12
    gl.xlabel_style = {'size': tick_fontsize}
    gl.ylabel_style = {'size': tick_fontsize}

    _raise_gridlines(gl, z=10)


    dlon = np.diff(lon).mean() if lon.size > 1 else 0.1
    dlat = np.diff(lat).mean() if lat.size > 1 else 0.1
    lon_edges = np.concatenate(([lon[0]-0.5*dlon], 0.5*(lon[:-1]+lon[1:]), [lon[-1]+0.5*dlon]))
    lat_edges = np.concatenate(([lat[0]-0.5*dlat], 0.5*(lat[:-1]+lat[1:]), [lat[-1]+0.5*dlat]))
    LON, LAT = np.meshgrid(lon_edges, lat_edges)

    vmax = np.nanpercentile(np.abs(field[np.isfinite(field)]), 98) if np.isfinite(field).any() else 1.0
    pcm = ax.pcolormesh(
        LON, LAT, 10000*field,
        shading="auto",
        cmap=cmap,
        vmin=-600,
        vmax=600,
        transform=ccrs.PlateCarree()
    )

    cb = fig.colorbar(pcm, ax=ax, shrink=0.6, extend="both")
    cb.set_label("ΔEKE = SWOT - DUACS [cm² s⁻²]", fontsize=12)
    cb.ax.tick_params(labelsize=12)

    ax.set_title(title)

    if savefig is not None:
        fig.savefig(savefig, dpi=DPI)
        print(f"[OK] saved: {savefig}")
    plt.close(fig)

# =============================================================================
# main
# =============================================================================

def main():
    lon_edges, lat_edges, lon_fine, lat_fine = make_grid(
        lon_min=LON_MIN, lon_max=LON_MAX,
        lat_min=LAT_MIN, lat_max=LAT_MAX,
        res=RES_FINE
    )
    ny, nx = lat_fine.size, lon_fine.size

    l3_by_cycle_day = index_l3_by_cycle_and_day(L3_ROOT)
    duacs_by_day = index_duacs_by_day(DUACS_CACHE)

    cycle_maps = {}
    cycle_to_season = {}
    summary_rows = []

    for cycle_id in sorted(l3_by_cycle_day.keys()):
        print(f"\n=== Processing cycle {cycle_id:03d} ===")
        day_to_l3files = l3_by_cycle_day[cycle_id]

        diff_cycle, diff_cnt, used_days = process_cycle_daily_differences(
            cycle_id=cycle_id,
            day_to_l3files=day_to_l3files,
            duacs_by_day=duacs_by_day,
            lon_edges=lon_edges,
            lat_edges=lat_edges,
            lon_fine=lon_fine,
            lat_fine=lat_fine,
            ny=ny,
            nx=nx,
            min_count=MIN_COUNT_SWOT_DAY
        )

        if not np.isfinite(diff_cycle).any():
            print(f"[WARN] no valid diff for cycle {cycle_id:03d}")
            continue

        season = assign_cycle_to_season(used_days)
        if season is None:
            print(f"[INFO] cycle {cycle_id:03d} not assigned to winter/summer")
            continue

        cycle_maps[cycle_id] = diff_cycle
        cycle_to_season[cycle_id] = season

        if SAVE_NPZ:
            np.savez_compressed(
                OUTDIR / f"diff_cycle_{cycle_id:03d}.npz",
                lon=lon_fine,
                lat=lat_fine,
                diff_cycle=diff_cycle,
                diff_cnt=diff_cnt
            )

        summary_rows.append({
            "cycle": cycle_id,
            "season": season,
            "used_days": len(used_days),
            "mid_date": cycle_mid_date(used_days).isoformat() if used_days else "",
            "mean_diff": float(np.nanmean(diff_cycle)),
            "std_diff": float(np.nanstd(diff_cycle)),
            "n_valid_pixels": int(np.isfinite(diff_cycle).sum()),
        })

    seasonal_maps = seasonal_composite_from_cycles(cycle_maps, cycle_to_season)

    for season, field in seasonal_maps.items():
        if field is None:
            print(f"[WARN] no field for {season}")
            continue

        if SAVE_NPZ:
            np.savez_compressed(
                OUTDIR / f"{season}_diff_map.npz",
                lon=lon_fine,
                lat=lat_fine,
                diff=field
            )

        if SAVE_FIG:
            plot_diff_map(
                lon=lon_fine,
                lat=lat_fine,
                field=field,
                title="",
                savefig=OUTDIR / f"{season}_diff_map.png"
            )

    csv_path = OUTDIR / "cycle_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["cycle", "season", "used_days", "mid_date", "mean_diff", "std_diff", "n_valid_pixels"]
        )
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    print(f"\n[OK] summary saved: {csv_path}")
    print("[OK] done")

if __name__ == "__main__":
    main()