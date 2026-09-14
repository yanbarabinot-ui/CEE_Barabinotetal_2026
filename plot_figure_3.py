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
import cmocean
from matplotlib.patches import Rectangle
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

# Okabe–Ito colour-blind-safe palette used for the bar plots.
BAR_BLUE = "#0072B2"
BAR_ORANGE = "#E69F00"
BAR_TITLE_FONTSIZE = 15
BAR_LABEL_FONTSIZE = 13
BAR_TICK_FONTSIZE = 12
BAR_LEGEND_FONTSIZE = 12
BAR_ANNOTATION_FONTSIZE = 12
BAR_PANEL_FONTSIZE = 20

FN_L3_DATE = re.compile(r"_(\d{8})T\d{6}_")
FN_DUACS_DATE = re.compile(r"(\d{8})")

SEASONS = {
    "winter": {12, 1, 2},
    "summer": {6, 7, 8},
}

# Rectangular regions used both on panel a and for the regional statistics.
# Bounds are (lon_min, lon_max, lat_min, lat_max) in degrees.
REGIONS = [
    {
        "short_name": "Alboran",
        "map_name": "Alboran Sea",
        "bounds": (-5.8, -1.0, 35.0, 37.3),
    },
    {
        "short_name": "Algerian",
        "map_name": "Algerian basin",
        "bounds": (-1.0, 10.0, 35.0, 39.5),
    },
    {
        "short_name": "Lev.",
        "map_name": "Levantine basin",
        "bounds": (22.0, 36.0, 30.0, 36.7),
    },
]

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
    # SWOT, DUACS and their difference are accumulated over exactly the same
    # daily SWOT-covered pixels. This guarantees that the maps and bar plots
    # use the same matched sampling period and spatial support.
    swot_sum = np.zeros((ny, nx), dtype=np.float64)
    duacs_sum = np.zeros((ny, nx), dtype=np.float64)
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

        swot_sum[valid] += swot_day[valid]
        duacs_sum[valid] += duacs_on_fine[valid]
        diff_sum[valid] += diff_day[valid]
        diff_cnt[valid] += 1
        used_days.append(day)

    swot_cycle = np.full((ny, nx), np.nan, dtype=np.float64)
    duacs_cycle = np.full((ny, nx), np.nan, dtype=np.float64)
    diff_cycle = np.full((ny, nx), np.nan, dtype=np.float64)

    valid = diff_cnt > 0
    swot_cycle[valid] = swot_sum[valid] / diff_cnt[valid]
    duacs_cycle[valid] = duacs_sum[valid] / diff_cnt[valid]
    diff_cycle[valid] = diff_sum[valid] / diff_cnt[valid]

    return swot_cycle, duacs_cycle, diff_cycle, diff_cnt, used_days

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

def plot_diff_map(lon, lat, field, title, savefig=None, cmap=cmocean.cm.balance, show_regions=False):
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
        vmin=-300,
        vmax=300,
        transform=ccrs.PlateCarree()
    )
    """
    # ΔEKE = 0 isoline.
    ax.contour(
        lon, lat, 10000*field,
        levels=[0.0],
        colors="black",
        linewidths=0.6,
        alpha=0.75,
        transform=ccrs.PlateCarree(),
        zorder=6,
    )
    """
    if show_regions:
        add_region_boxes(ax)

    cb = fig.colorbar(pcm, ax=ax, shrink=0.6, extend="both")
    cb.set_label("ΔEKE = SWOT - DUACS [cm² s⁻²]", fontsize=12)
    cb.ax.tick_params(labelsize=12)

    ax.set_title(title)

    if savefig is not None:
        fig.savefig(savefig, dpi=DPI)
        print(f"[OK] saved: {savefig}")
    plt.close(fig)


def add_region_boxes(ax):
    """Draw the three regions of interest on the first map panel."""
    for region in REGIONS:
        lon0, lon1, lat0, lat1 = region["bounds"]
        rect = Rectangle(
            (lon0, lat0),
            lon1 - lon0,
            lat1 - lat0,
            fill=False,
            edgecolor="black",
            linewidth=1.2,
            linestyle="--",
            transform=ccrs.PlateCarree(),
            zorder=12,
        )
        ax.add_patch(rect)
        ax.text(
            lon0 + 0.18,
            lat1 - 0.22,
            region["map_name"],
            transform=ccrs.PlateCarree(),
            ha="left",
            va="top",
            fontsize=9,
            fontweight="bold",
            zorder=13,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
        )

        # Label for the Strait of Gibraltar
    ax.text(
        -5.65,
        34,
        "Strait of Gibraltar",
        transform=ccrs.PlateCarree(),
        ha="left",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        zorder=14,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.4},
    )

def make_region_mask(lon, lat, bounds=None):
    LON, LAT = np.meshgrid(lon, lat)

    if bounds is None:
        flat = np.isfinite(LON.ravel()) & np.isfinite(LAT.ravel())
        inside = np.zeros(LON.size, dtype=bool)
        inside[flat] = contains_xy(ROI_POLY, LON.ravel()[flat], LAT.ravel()[flat])
        return inside.reshape(LON.shape)

    lon0, lon1, lat0, lat1 = bounds
    return (
        (LON >= lon0) & (LON <= lon1) &
        (LAT >= lat0) & (LAT <= lat1)
    )


def masked_mean(field, mask):
    valid = mask & np.isfinite(field)
    if not np.any(valid):
        return np.nan
    return float(np.mean(field[valid]))


def finite_sem(values):
    """Standard error of finite values. Returns NaN if fewer than two values exist."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    return float(np.nanstd(values, ddof=1) / np.sqrt(values.size))


def regional_cycle_mean_series(cycle_maps, cycle_to_season, season, mask, scale=1.0):
    """Regional mean for each independent cycle assigned to a given season."""
    if cycle_maps is None or cycle_to_season is None:
        return np.array([], dtype=float)

    values = []
    for cycle_id in sorted(cycle_maps.keys()):
        if cycle_to_season.get(cycle_id) != season:
            continue
        value = masked_mean(cycle_maps[cycle_id], mask)
        if np.isfinite(value):
            values.append(scale * value)

    return np.asarray(values, dtype=float)


def propagated_amplification_error_percent(jja_mean, djf_mean, jja_error, djf_error):
    """Error propagation for 100 * (DJF - JJA) / JJA.

    The uncertainty is propagated from the uncertainty of the two regional
    seasonal means, not from pixel-wise percentages. This avoids pathological
    values when individual pixels have very small JJA EKE.
    """
    if (
        not np.isfinite(jja_mean) or
        not np.isfinite(djf_mean) or
        not np.isfinite(jja_error) or
        not np.isfinite(djf_error) or
        jja_mean == 0.0
    ):
        return np.nan

    return float(
        100.0 * np.sqrt(
            (djf_error / jja_mean) ** 2 +
            ((djf_mean * jja_error) / (jja_mean ** 2)) ** 2
        )
    )


def compute_regional_metrics(
    lon,
    lat,
    seasonal_diff_maps,
    seasonal_swot_maps,
    seasonal_duacs_maps,
    cycle_diff_maps=None,
    cycle_swot_maps=None,
    cycle_duacs_maps=None,
    cycle_to_season=None,
):
    """Compute the diagnostics displayed in panels c and d.

    Error bars are estimated from cycle-to-cycle variability of regional means.
    This gives an uncertainty on the seasonal/regional mean itself and avoids
    the very large values produced by pixel-wise percentage ratios.
    """
    region_specs = REGIONS + [
        {
            "short_name": "Basin mean",
            "map_name": "Mediterranean basin mean",
            "bounds": None,
        }
    ]

    metrics = []
    for region in region_specs:
        mask = make_region_mask(lon, lat, region["bounds"])

        # Individual observations for panel c are the regional mean ΔEKE values
        # from each independent 21-day SWOT cycle assigned to JJA or DJF.
        # The bar height and SEM are computed from exactly these displayed points.
        diff_jja_series = regional_cycle_mean_series(
            cycle_diff_maps, cycle_to_season, "summer", mask, scale=10000.0
        )
        diff_djf_series = regional_cycle_mean_series(
            cycle_diff_maps, cycle_to_season, "winter", mask, scale=10000.0
        )

        diff_jja = (
            float(np.nanmean(diff_jja_series))
            if diff_jja_series.size else
            10000.0 * masked_mean(seasonal_diff_maps["summer"], mask)
        )
        diff_djf = (
            float(np.nanmean(diff_djf_series))
            if diff_djf_series.size else
            10000.0 * masked_mean(seasonal_diff_maps["winter"], mask)
        )
        diff_jja_err = finite_sem(diff_jja_series)
        diff_djf_err = finite_sem(diff_djf_series)

        # Use one common spatial support for both products and both seasons in
        # the amplification diagnostic.
        common = (
            mask &
            np.isfinite(seasonal_swot_maps["summer"]) &
            np.isfinite(seasonal_swot_maps["winter"]) &
            np.isfinite(seasonal_duacs_maps["summer"]) &
            np.isfinite(seasonal_duacs_maps["winter"])
        )

        swot_jja = 10000.0 * masked_mean(seasonal_swot_maps["summer"], common)
        swot_djf = 10000.0 * masked_mean(seasonal_swot_maps["winter"], common)
        duacs_jja = 10000.0 * masked_mean(seasonal_duacs_maps["summer"], common)
        duacs_djf = 10000.0 * masked_mean(seasonal_duacs_maps["winter"], common)

        swot_jja_series = regional_cycle_mean_series(
            cycle_swot_maps, cycle_to_season, "summer", common, scale=10000.0
        )
        swot_djf_series = regional_cycle_mean_series(
            cycle_swot_maps, cycle_to_season, "winter", common, scale=10000.0
        )
        duacs_jja_series = regional_cycle_mean_series(
            cycle_duacs_maps, cycle_to_season, "summer", common, scale=10000.0
        )
        duacs_djf_series = regional_cycle_mean_series(
            cycle_duacs_maps, cycle_to_season, "winter", common, scale=10000.0
        )

        swot_jja_err = finite_sem(swot_jja_series)
        swot_djf_err = finite_sem(swot_djf_series)
        duacs_jja_err = finite_sem(duacs_jja_series)
        duacs_djf_err = finite_sem(duacs_djf_series)

        swot_amp = (
            100.0 * (swot_djf - swot_jja) / swot_jja
            if np.isfinite(swot_jja) and swot_jja != 0.0 else np.nan
        )
        duacs_amp = (
            100.0 * (duacs_djf - duacs_jja) / duacs_jja
            if np.isfinite(duacs_jja) and duacs_jja != 0.0 else np.nan
        )

        swot_amp_err = propagated_amplification_error_percent(
            swot_jja, swot_djf, swot_jja_err, swot_djf_err
        )
        duacs_amp_err = propagated_amplification_error_percent(
            duacs_jja, duacs_djf, duacs_jja_err, duacs_djf_err
        )

        metrics.append({
            "region": region["short_name"],
            "diff_jja_cm2_s2": diff_jja,
            "diff_djf_cm2_s2": diff_djf,
            "diff_jja_error_cm2_s2": diff_jja_err,
            "diff_djf_error_cm2_s2": diff_djf_err,
            "_diff_jja_cycle_values_cm2_s2": diff_jja_series,
            "_diff_djf_cycle_values_cm2_s2": diff_djf_series,
            "swot_jja_cm2_s2": swot_jja,
            "swot_djf_cm2_s2": swot_djf,
            "duacs_jja_cm2_s2": duacs_jja,
            "duacs_djf_cm2_s2": duacs_djf,
            "swot_jja_error_cm2_s2": swot_jja_err,
            "swot_djf_error_cm2_s2": swot_djf_err,
            "duacs_jja_error_cm2_s2": duacs_jja_err,
            "duacs_djf_error_cm2_s2": duacs_djf_err,
            "swot_amplification_percent": swot_amp,
            "duacs_amplification_percent": duacs_amp,
            "swot_amplification_error_percent": swot_amp_err,
            "duacs_amplification_error_percent": duacs_amp_err,
            "n_jja_cycles": int(diff_jja_series.size),
            "n_djf_cycles": int(diff_djf_series.size),
            "n_common_pixels": int(np.count_nonzero(common)),
        })

    return metrics


def _clean_yerr(yerr):
    """Matplotlib-friendly yerr: finite, non-negative, zero when unavailable."""
    yerr = np.asarray(yerr, dtype=float)
    return np.where(np.isfinite(yerr) & (yerr >= 0.0), yerr, 0.0)


def _plot_individual_cycle_points(ax, x_positions, metrics, key, color, width):
    """Overlay cycle-level regional means on a bar plot with deterministic jitter."""
    for xpos, row in zip(x_positions, metrics):
        values = np.asarray(row.get(key, []), dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue

        # Deterministic horizontal spread: no random state and no visual implication
        # of an additional variable. Keep all points within the corresponding bar.
        if values.size == 1:
            jitter = np.array([0.0])
        else:
            jitter = np.linspace(-0.28 * width, 0.28 * width, values.size)

        ax.scatter(
            xpos + jitter,
            values,
            s=24,
            facecolor=color,
            edgecolor="black",
            linewidth=0.55,
            alpha=0.9,
            zorder=5,
            clip_on=False,
        )


def plot_regional_bar_panels(metrics, savefig_c, savefig_d):
    """Save panels c and d as two independent publication-ready figures."""
    labels = [row["region"] for row in metrics]
    x = np.arange(len(labels), dtype=float)
    width = 0.34

    # -------------------------------------------------------------------------
    # Panel c: regional SWOT-DUACS EKE difference
    # -------------------------------------------------------------------------
    diff_jja = np.array([row["diff_jja_cm2_s2"] for row in metrics])
    diff_djf = np.array([row["diff_djf_cm2_s2"] for row in metrics])
    diff_jja_err = np.array([row["diff_jja_error_cm2_s2"] for row in metrics])
    diff_djf_err = np.array([row["diff_djf_error_cm2_s2"] for row in metrics])
    error_kw = {"elinewidth": 1.1, "capthick": 1.1}

    fig_c, ax_c = plt.subplots(figsize=(5, 4), constrained_layout=True)
    fig_c.patch.set_facecolor("white")
    ax_c.set_facecolor("white")

    bars_jja = ax_c.bar(
        x - width / 2.0,
        diff_jja,
        width,
        label="JJA",
        color=BAR_BLUE,
        edgecolor="black",
        linewidth=0.5,
        yerr=_clean_yerr(diff_jja_err),
        capsize=3,
        error_kw=error_kw,
    )
    bars_djf = ax_c.bar(
        x + width / 2.0,
        diff_djf,
        width,
        label="DJF",
        color=BAR_ORANGE,
        edgecolor="black",
        linewidth=0.5,
        yerr=_clean_yerr(diff_djf_err),
        capsize=3,
        error_kw=error_kw,
    )

    ax_c.axhline(0.0, color="black", linewidth=0.8)
    ax_c.set_title(
        "Regional SWOT–DUACS EKE difference",
        fontsize=BAR_TITLE_FONTSIZE,
        fontweight="bold",
        pad=11,
    )
    ax_c.set_ylabel("ΔEKE [cm² s⁻²]", fontsize=BAR_LABEL_FONTSIZE)
    ax_c.set_xticks(x, labels, fontsize=BAR_TICK_FONTSIZE)
    ax_c.tick_params(axis="y", labelsize=BAR_TICK_FONTSIZE)
    ax_c.grid(axis="y", linestyle="--", alpha=0.45)
    ax_c.set_axisbelow(True)
    ax_c.legend(frameon=True, fontsize=BAR_LEGEND_FONTSIZE)
    ax_c.margins(y=0.25)
    _plot_individual_cycle_points(
        ax_c, x - width / 2.0, metrics,
        "_diff_jja_cycle_values_cm2_s2", BAR_BLUE, width
    )
    _plot_individual_cycle_points(
        ax_c, x + width / 2.0, metrics,
        "_diff_djf_cycle_values_cm2_s2", BAR_ORANGE, width
    )
    
    _annotate_bars(ax_c, bars_jja, diff_jja, diff_jja_err, boxed=True)
    _annotate_bars(ax_c, bars_djf, diff_djf, diff_djf_err, boxed=True)

    fig_c.savefig(savefig_c, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"[OK] saved: {savefig_c}")
    plt.close(fig_c)

    # -------------------------------------------------------------------------
    # Panel d: seasonal amplification
    # -------------------------------------------------------------------------
    swot_amp = np.array([row["swot_amplification_percent"] for row in metrics])
    duacs_amp = np.array([row["duacs_amplification_percent"] for row in metrics])
    swot_amp_err = np.array([row["swot_amplification_error_percent"] for row in metrics])
    duacs_amp_err = np.array([row["duacs_amplification_error_percent"] for row in metrics])
    error_kw = {"elinewidth": 1.1, "capthick": 1.1}

    fig_d, ax_d = plt.subplots(figsize=(5, 4), constrained_layout=True)
    fig_d.patch.set_facecolor("white")
    ax_d.set_facecolor("white")

    bars_swot = ax_d.bar(
        x - width / 2.0,
        swot_amp,
        width,
        label="SWOT L3 v3",
        color="#009E73",
        edgecolor="black",
        linewidth=0.5,
        yerr=_clean_yerr(swot_amp_err),
        capsize=3,
        error_kw=error_kw,
    )
    bars_duacs = ax_d.bar(
        x + width / 2.0,
        duacs_amp,
        width,
        label="DUACS swot-like",
        color="#CC79A7",
        edgecolor="black",
        linewidth=0.5,
        yerr=_clean_yerr(duacs_amp_err),
        capsize=3,
        error_kw=error_kw,
    )

    ax_d.axhline(0.0, color="black", linewidth=0.8)
    ax_d.set_title(
        "Regional seasonal amplification",
        fontsize=BAR_TITLE_FONTSIZE,
        fontweight="bold",
        pad=11,
    )
    ax_d.set_ylabel("(DJF − JJA) / JJA [%]", fontsize=BAR_LABEL_FONTSIZE)
    ax_d.set_xticks(x, labels, fontsize=BAR_TICK_FONTSIZE)
    ax_d.tick_params(axis="y", labelsize=BAR_TICK_FONTSIZE)
    ax_d.grid(axis="y", linestyle="--", alpha=0.45)
    ax_d.set_axisbelow(True)
    ax_d.legend(frameon=True, fontsize=BAR_LEGEND_FONTSIZE)
    ax_d.margins(y=0.28)
    _annotate_bars(ax_d, bars_swot, swot_amp, swot_amp_err, boxed=True)
    _annotate_bars(ax_d, bars_duacs, duacs_amp, duacs_amp_err, boxed=True)

    fig_d.savefig(savefig_d, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"[OK] saved: {savefig_d}")
    plt.close(fig_d)


def _configure_map_axis(ax, show_bottom_labels):
    import matplotlib.ticker as mticker
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    land = cfeature.NaturalEarthFeature("physical", "land", "50m")
    ax.add_feature(
        land,
        facecolor="lightgray",
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
    )
    ax.coastlines(resolution="50m", color="black", linewidth=0.6, zorder=4)
    ax.set_extent([-6, 36, 30, 46], crs=ccrs.PlateCarree())

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = show_bottom_labels
    gl.xlocator = mticker.FixedLocator(np.arange(-6, 37, 6))
    gl.ylocator = mticker.FixedLocator(np.arange(30, 47, 2))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 11}
    gl.ylabel_style = {"size": 11}
    _raise_gridlines(gl, z=10)


def _plot_map_panel(
    fig,
    ax,
    lon,
    lat,
    field,
    panel_label,
    season_label,
    show_bottom_labels,
    show_regions,
    cmap=cmocean.cm.balance,
):
    _configure_map_axis(ax, show_bottom_labels=show_bottom_labels)

    lon_edges = compute_edges(lon)
    lat_edges = compute_edges(lat)
    LONe, LATe = np.meshgrid(lon_edges, lat_edges)

    pcm = ax.pcolormesh(
        LONe,
        LATe,
        10000.0 * field,
        shading="auto",
        cmap=cmap,
        vmin=-300,
        vmax=300,
        transform=ccrs.PlateCarree(),
        zorder=1,
    )

    ax.contour(
        lon,
        lat,
        10000.0 * field,
        levels=[0.0],
        colors="black",
        linewidths=0.6,
        alpha=0.75,
        transform=ccrs.PlateCarree(),
        zorder=6,
    )

    if show_regions:
        add_region_boxes(ax)

    ax.text(
        0.015,
        0.965,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
        zorder=20,
    )
    ax.text(
        0.015,
        0.025,
        season_label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=17,
        fontweight="bold",
        zorder=20,
    )

    cb = fig.colorbar(pcm, ax=ax, shrink=0.78, pad=0.025, extend="both")
    cb.set_label("ΔEKE = SWOT - DUACS [cm² s⁻²]", fontsize=11)
    cb.ax.tick_params(labelsize=10)


def _annotate_bars(ax, bars, values, errors=None, boxed=False):
    if errors is None:
        errors = np.zeros_like(values, dtype=float)

    for bar, value, error in zip(bars, values, errors):
        if not np.isfinite(value):
            continue

        error = error if np.isfinite(error) else 0.0
        y_text = value + error if value >= 0 else value - error
        va = "bottom" if value >= 0 else "top"
        offset = 3 if value >= 0 else -3

        bbox = (
            dict(
                boxstyle="round,pad=0.12",
                facecolor="white",
                edgecolor="black",
                linewidth=0.5,
                alpha=0.7,
            )
            if boxed else None
        )

        ax.annotate(
            f"{value:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2.0, y_text),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=BAR_ANNOTATION_FONTSIZE,
            bbox=bbox,
            zorder=30,
            clip_on=False,
        )


def plot_four_panel_figure(
    lon,
    lat,
    seasonal_diff_maps,
    seasonal_swot_maps,
    seasonal_duacs_maps,
    savefig,
    cycle_diff_maps=None,
    cycle_swot_maps=None,
    cycle_duacs_maps=None,
    cycle_to_season=None,
    cmap=cmocean.cm.balance,
):
    metrics = compute_regional_metrics(
        lon=lon,
        lat=lat,
        seasonal_diff_maps=seasonal_diff_maps,
        seasonal_swot_maps=seasonal_swot_maps,
        seasonal_duacs_maps=seasonal_duacs_maps,
        cycle_diff_maps=cycle_diff_maps,
        cycle_swot_maps=cycle_swot_maps,
        cycle_duacs_maps=cycle_duacs_maps,
        cycle_to_season=cycle_to_season,
    )

    fig = plt.figure(figsize=(14, 14.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.78])

    proj_map = ccrs.Mercator()
    ax_a = fig.add_subplot(gs[0, :], projection=proj_map)
    ax_b = fig.add_subplot(gs[1, :], projection=proj_map)
    ax_c = fig.add_subplot(gs[2, 0])
    ax_d = fig.add_subplot(gs[2, 1])

    _plot_map_panel(
        fig=fig,
        ax=ax_a,
        lon=lon,
        lat=lat,
        field=seasonal_diff_maps["summer"],
        panel_label="a",
        season_label="JJA",
        show_bottom_labels=False,
        show_regions=True,
        cmap=cmap,
    )
    _plot_map_panel(
        fig=fig,
        ax=ax_b,
        lon=lon,
        lat=lat,
        field=seasonal_diff_maps["winter"],
        panel_label="b",
        season_label="DJF",
        show_bottom_labels=True,
        show_regions=False,
        cmap=cmap,
    )

    labels = [row["region"] for row in metrics]
    x = np.arange(len(labels), dtype=float)
    width = 0.34

    diff_jja = np.array([row["diff_jja_cm2_s2"] for row in metrics])
    diff_djf = np.array([row["diff_djf_cm2_s2"] for row in metrics])
    diff_jja_err = np.array([row["diff_jja_error_cm2_s2"] for row in metrics])
    diff_djf_err = np.array([row["diff_djf_error_cm2_s2"] for row in metrics])
    error_kw = {"elinewidth": 1.1, "capthick": 1.1}
    bars_jja = ax_c.bar(
        x - width / 2.0, diff_jja, width, label="JJA",
        color=BAR_BLUE, edgecolor="black", linewidth=0.4,
        yerr=_clean_yerr(diff_jja_err), capsize=3, error_kw=error_kw
    )
    bars_djf = ax_c.bar(
        x + width / 2.0, diff_djf, width, label="DJF",
        color=BAR_ORANGE, edgecolor="black", linewidth=0.4,
        yerr=_clean_yerr(diff_djf_err), capsize=3, error_kw=error_kw
    )
    ax_c.axhline(0.0, color="black", linewidth=0.7)
    ax_c.set_title(
        "Regional SWOT–DUACS EKE difference",
        fontsize=BAR_TITLE_FONTSIZE, fontweight="bold", pad=10
    )
    ax_c.set_ylabel("ΔEKE [cm² s⁻²]", fontsize=BAR_LABEL_FONTSIZE)
    ax_c.set_xticks(x, labels, fontsize=BAR_TICK_FONTSIZE)
    ax_c.tick_params(axis="y", labelsize=BAR_TICK_FONTSIZE)
    ax_c.grid(axis="y", linestyle="--", alpha=0.45)
    ax_c.set_axisbelow(True)
    ax_c.legend(frameon=True, fontsize=BAR_LEGEND_FONTSIZE)
    ax_c.margins(y=0.25)
    ax_c.text(
        0.02, 0.96, "c", transform=ax_c.transAxes,
        ha="left", va="top", fontsize=BAR_PANEL_FONTSIZE, fontweight="bold"
    )
    _plot_individual_cycle_points(
        ax_c, x - width / 2.0, metrics,
        "_diff_jja_cycle_values_cm2_s2", BAR_BLUE, width
    )
    _plot_individual_cycle_points(
        ax_c, x + width / 2.0, metrics,
        "_diff_djf_cycle_values_cm2_s2", BAR_ORANGE, width
    )
    _annotate_bars(ax_c, bars_jja, diff_jja, diff_jja_err, boxed=True)
    _annotate_bars(ax_c, bars_djf, diff_djf, diff_djf_err, boxed=True)

    swot_amp = np.array([row["swot_amplification_percent"] for row in metrics])
    duacs_amp = np.array([row["duacs_amplification_percent"] for row in metrics])
    swot_amp_err = np.array([row["swot_amplification_error_percent"] for row in metrics])
    duacs_amp_err = np.array([row["duacs_amplification_error_percent"] for row in metrics])
    error_kw = {"elinewidth": 1.1, "capthick": 1.1}
    bars_swot = ax_d.bar(
        x - width / 2.0, swot_amp, width, label="SWOT L3 v3",
        color=BAR_BLUE, edgecolor="black", linewidth=0.4,
        yerr=_clean_yerr(swot_amp_err), capsize=3, error_kw=error_kw
    )
    bars_duacs = ax_d.bar(
        x + width / 2.0, duacs_amp, width, label="DUACS swot-like",
        color=BAR_ORANGE, edgecolor="black", linewidth=0.4,
        yerr=_clean_yerr(duacs_amp_err), capsize=3, error_kw=error_kw
    )
    ax_d.axhline(0.0, color="black", linewidth=0.7)
    ax_d.set_title(
        "Regional seasonal amplification",
        fontsize=BAR_TITLE_FONTSIZE, fontweight="bold", pad=10
    )
    ax_d.set_ylabel("(DJF − JJA) / JJA [%]", fontsize=BAR_LABEL_FONTSIZE)
    ax_d.set_xticks(x, labels, fontsize=BAR_TICK_FONTSIZE)
    ax_d.tick_params(axis="y", labelsize=BAR_TICK_FONTSIZE)
    ax_d.grid(axis="y", linestyle="--", alpha=0.45)
    ax_d.set_axisbelow(True)
    ax_d.legend(frameon=True, fontsize=BAR_LEGEND_FONTSIZE)
    ax_d.margins(y=0.26)
    ax_d.text(
        0.02, 0.96, "d", transform=ax_d.transAxes,
        ha="left", va="top", fontsize=BAR_PANEL_FONTSIZE, fontweight="bold"
    )
    _annotate_bars(ax_d, bars_swot, swot_amp, swot_amp_err, boxed=True)
    _annotate_bars(ax_d, bars_duacs, duacs_amp, duacs_amp_err, boxed=True)

    fig.savefig(savefig, dpi=DPI, bbox_inches="tight")
    print(f"[OK] saved: {savefig}")
    plt.close(fig)

    return metrics


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
    cycle_swot_maps = {}
    cycle_duacs_maps = {}
    cycle_to_season = {}
    summary_rows = []

    for cycle_id in sorted(l3_by_cycle_day.keys()):
        print(f"\n=== Processing cycle {cycle_id:03d} ===")
        day_to_l3files = l3_by_cycle_day[cycle_id]

        swot_cycle, duacs_cycle, diff_cycle, diff_cnt, used_days = process_cycle_daily_differences(
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
        cycle_swot_maps[cycle_id] = swot_cycle
        cycle_duacs_maps[cycle_id] = duacs_cycle
        cycle_to_season[cycle_id] = season

        if SAVE_NPZ:
            np.savez_compressed(
                OUTDIR / f"diff_cycle_{cycle_id:03d}.npz",
                lon=lon_fine,
                lat=lat_fine,
                swot_cycle=swot_cycle,
                duacs_cycle=duacs_cycle,
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
    seasonal_swot_maps = seasonal_composite_from_cycles(cycle_swot_maps, cycle_to_season)
    seasonal_duacs_maps = seasonal_composite_from_cycles(cycle_duacs_maps, cycle_to_season)

    for season, field in seasonal_maps.items():
        if field is None:
            print(f"[WARN] no field for {season}")
            continue

        if SAVE_NPZ:
            np.savez_compressed(
                OUTDIR / f"{season}_diff_map.npz",
                lon=lon_fine,
                lat=lat_fine,
                swot=seasonal_swot_maps[season],
                duacs=seasonal_duacs_maps[season],
                diff=field
            )

        if SAVE_FIG:
            plot_diff_map(
                lon=lon_fine,
                lat=lat_fine,
                field=field,
                title="",
                savefig=OUTDIR / f"{season}_diff_map.png",
                show_regions=(season == "summer")
            )


    regional_metrics = []
    required_maps = (
        seasonal_maps.get("summer") is not None and
        seasonal_maps.get("winter") is not None and
        seasonal_swot_maps.get("summer") is not None and
        seasonal_swot_maps.get("winter") is not None and
        seasonal_duacs_maps.get("summer") is not None and
        seasonal_duacs_maps.get("winter") is not None
    )

    if SAVE_FIG and required_maps:
        regional_metrics = plot_four_panel_figure(
            lon=lon_fine,
            lat=lat_fine,
            seasonal_diff_maps=seasonal_maps,
            seasonal_swot_maps=seasonal_swot_maps,
            seasonal_duacs_maps=seasonal_duacs_maps,
            savefig=OUTDIR / "seasonal_diff_maps_and_regional_bars.png",
            cycle_diff_maps=cycle_maps,
            cycle_swot_maps=cycle_swot_maps,
            cycle_duacs_maps=cycle_duacs_maps,
            cycle_to_season=cycle_to_season,
        )
        plot_regional_bar_panels(
            metrics=regional_metrics,
            savefig_c=OUTDIR / "panel_c_regional_eke_difference.png",
            savefig_d=OUTDIR / "panel_d_regional_seasonal_amplification.png",
        )
    elif SAVE_FIG:
        print("[WARN] the four-panel figure requires both JJA and DJF composites")

    if required_maps and not regional_metrics:
        regional_metrics = compute_regional_metrics(
            lon=lon_fine,
            lat=lat_fine,
            seasonal_diff_maps=seasonal_maps,
            seasonal_swot_maps=seasonal_swot_maps,
            seasonal_duacs_maps=seasonal_duacs_maps,
            cycle_diff_maps=cycle_maps,
            cycle_swot_maps=cycle_swot_maps,
            cycle_duacs_maps=cycle_duacs_maps,
            cycle_to_season=cycle_to_season,
        )

    if regional_metrics:
        regional_csv_path = OUTDIR / "regional_metrics.csv"
        with regional_csv_path.open("w", newline="") as f:
            # Private keys (prefixed with _) contain cycle-level arrays used only
            # to draw individual points and are intentionally omitted from CSV.
            csv_fieldnames = [
                key for key in regional_metrics[0].keys()
                if not key.startswith("_")
            ]
            w = csv.DictWriter(f, fieldnames=csv_fieldnames, extrasaction="ignore")
            w.writeheader()
            for row in regional_metrics:
                w.writerow(row)
        print(f"[OK] regional metrics saved: {regional_csv_path}")

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