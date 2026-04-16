#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 16:53:22 2025

@author: yan
"""

"""
Joint PDF contours (S/f, zeta/f) for multiple cycles (1..36) on ONE figure.
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable

import shapely
from shapely.geometry import box
from shapely import contains_xy
import pandas as pd
from scipy.ndimage import zoom

# ---------------- Constants ----------------
EARTH_RADIUS_M = 6371000.0
OMEGA = 7.292115e-5  # rad/s
plt.rcParams.update({"font.size": 12})


# ---------------- Mediterranean ROI ----------------
def build_med_mask():
    med_bbox   = box(-6, 30, 36, 46)
    black_sea  = box(27, 41, 42, 47)
    bay_biscay = box(-6, 43, -1, 46)
    roi = shapely.difference(med_bbox, shapely.unary_union([black_sea, bay_biscay]))
    return shapely.buffer(roi, 0)

ROI_POLY = build_med_mask()

def to_m180_180(lon):
    return ((lon + 180) % 360) - 180

# ---------------- Grid & binning helpers ----------------
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
    dx_left[:, 0] = dx_right[:, 0]
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
    dy_up[0, :] = dy_down[0, :]
    dy_sum = dy_up + dy_down

    m_cy = A[2:, :] & A[:-2, :] & np.isfinite(dy_sum[1:-1, :]) & (dy_sum[1:-1, :] > 0)
    day[1:-1, :][m_cy] = (a[2:, :][m_cy] - a[:-2, :][m_cy]) / dy_sum[1:-1, :][m_cy]

    m_fy = A[1, :] & A[0, :] & np.isfinite(dy_down[0, :]) & (dy_down[0, :][...] > 0)
    day[0, :][m_fy] = (a[1, :][m_fy] - a[0, :][m_fy]) / dy_down[0, :][m_fy]

    m_by = A[-1, :] & A[-2, :] & np.isfinite(dy_up[-1, :]) & (dy_up[-1, :][...] > 0)
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

# ---------------- bathymetry helper (optional) ----------------
def _normalize_lon_to_180(lon):
    lon = np.asarray(lon)
    if np.nanmax(lon) > 180.0:
        lon = ((lon + 180.0) % 360.0) - 180.0
    return lon

def _build_topo_interpolator(topo_nc_path):
    with Dataset(topo_nc_path) as ds:
        topo_lon = np.array(ds.variables["lon"][:], dtype=np.float64)
        topo_lat = np.array(ds.variables["lat"][:], dtype=np.float64)
        topo     = np.array(ds.variables["topo"][:], dtype=np.float32)

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

        return ((1.0 - tx) * (1.0 - ty) * f00
              + tx         * (1.0 - ty) * f10
              + (1.0 - tx) * ty         * f01
              + tx         * ty         * f11).astype(np.float32)

    return interp

# ---------------- Read SWOT fields ----------------
def read_vars_from_nc(
    nc_path: Path,
    ssh_candidates=("ssha_filtered",),
    u_candidates=("ugosa_filtered",),
    v_candidates=("vgosa_filtered",),
    topo_interp=None,
    min_depth: Optional[float] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    with Dataset(nc_path, "r") as ds:

        vlon = "longitude" if "longitude" in ds.variables else ("lon" if "lon" in ds.variables else None)
        vlat = "latitude"  if "latitude"  in ds.variables else ("lat" if "lat" in ds.variables else None)
        if vlon is None or vlat is None:
            return None
        lon = np.array(ds.variables[vlon][:], dtype="float64")
        lat = np.array(ds.variables[vlat][:], dtype="float64")
        lon = to_m180_180(lon)

        ssh_var = next((nm for nm in ssh_candidates if nm in ds.variables), None)
        u_var   = next((nm for nm in u_candidates   if nm in ds.variables), None)
        v_var   = next((nm for nm in v_candidates   if nm in ds.variables), None)
        if ssh_var is None or u_var is None or v_var is None:
            return None

        SSH = ds.variables[ssh_var]
        U   = ds.variables[u_var]
        V   = ds.variables[v_var]

        SSH.set_auto_maskandscale(True)
        U.set_auto_maskandscale(True)
        V.set_auto_maskandscale(True)

        def read_nan(var):
            arr = var[:]
            return (arr.filled(np.nan) if np.ma.isMaskedArray(arr) else np.array(arr)).astype("float64")

        ssh = read_nan(SSH[0, ...]) if SSH.ndim == 3 else read_nan(SSH)
        u   = read_nan(U[0, ...])   if U.ndim   == 3 else read_nan(U)
        v   = read_nan(V[0, ...])   if V.ndim   == 3 else read_nan(V)

        def to_mps(arr, var):
            units = (getattr(var, "units", "") or "").lower().replace(" ", "")
            if units.startswith("cm/s") or units.replace(".", "").startswith("cms-1"):
                return arr / 100.0
            return arr

        u = to_mps(u, U)
        v = to_mps(v, V)

        if lon.ndim == 1 and lat.ndim == 1:
            LON, LAT = np.meshgrid(lon, lat)
        else:
            LON, LAT = lon, lat

        lonf = np.asarray(LON).ravel()
        latf = np.asarray(LAT).ravel()
        finite_xy = np.isfinite(lonf) & np.isfinite(latf)
        inside_flat = np.zeros_like(finite_xy, dtype=bool)
        inside_flat[finite_xy] = contains_xy(ROI_POLY, lonf[finite_xy], latf[finite_xy])
        inside = inside_flat.reshape(LON.shape)

        if (topo_interp is not None) and (min_depth is not None):
            flat_inside = inside.ravel()
            if np.any(flat_inside):
                topo_vals = topo_interp(lonf[flat_inside], latf[flat_inside])
                deep_ok_flat = np.isfinite(topo_vals) & (topo_vals <= float(min_depth))
                if not np.any(deep_ok_flat):
                    return None
                deep_ok = np.zeros_like(flat_inside, dtype=bool)
                deep_ok[flat_inside] = deep_ok_flat
                deep_ok = deep_ok.reshape(LON.shape)
                inside = inside & deep_ok
            else:
                return None

        if not np.any(inside):
            return None

        ssh = np.where(inside, ssh, np.nan)
        u   = np.where(inside, u,   np.nan)
        v   = np.where(inside, v,   np.nan)
        return LON, LAT, ssh, u, v

# ---------------- Strain & vort on bins ----------------
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
    Sf   = strain.ravel()
    Zf   = vort_norm.ravel()

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

# ---------------- Cycle -> month mapping from compare_ke_l3_l4.csv ----------------
def load_cycle_month_map(cycle_info_csv: Path) -> Dict[int, int]:
    df = pd.read_csv(cycle_info_csv, parse_dates=["date_min", "date_max"])
    df["cycle"] = df["cycle"].astype(int)
    df["date_median"] = df["date_min"] + (df["date_max"] - df["date_min"]) / 2.0
    df["month"] = df["date_median"].dt.month.astype(int)
    return dict(zip(df["cycle"].values, df["month"].values))

# ---------------- HDR contour level (enclosing a given mass) ----------------
def hdr_threshold(pdf2d: np.ndarray, dZ: float, dS: float, mass: float = 0.5) -> float:
    p = np.asarray(pdf2d, dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return np.nan

    # Sort densities descending
    p_sorted = np.sort(p)[::-1]
    # cumulative probability mass of bins included at each step
    cum = np.cumsum(p_sorted) * dZ * dS
    # Find smallest set reaching desired mass
    idx = np.searchsorted(cum, mass, side="left")
    if idx >= p_sorted.size:
        return p_sorted[-1]
    return p_sorted[idx]

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Overlay joint-PDF contours for cycles 1..36.")
    ap.add_argument("--l3_root", required=True, help="Root folder with cycle_XXX subfolders")
    ap.add_argument("--cycle-info", required=True, help="CSV with cycle/date_min/date_max (e.g. compare_ke_l3_l4.csv)")
    ap.add_argument("--cycle-start", type=int, default=1)
    ap.add_argument("--cycle-end", type=int, default=36)

    ap.add_argument("--res", type=float, default=0.025)
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--area-weight", action="store_true")

    ap.add_argument("--topo-file", default=None)
    ap.add_argument("--topo", type=float, default=None)

    ap.add_argument("--nbins-strain", type=int, default=200)
    ap.add_argument("--nbins-vort", type=int, default=200)
    ap.add_argument("--strain-max", type=float, default=2.0)
    ap.add_argument("--vort-max", type=float, default=2.0)

    ap.add_argument("--hdr-mass", type=float, default=0.95,
                    help="Probability mass enclosed by the contour (e.g. 0.5 for 50%% HDR)")
    ap.add_argument("--outfile", default=None)
    args = ap.parse_args()

    l3_root = Path(args.l3_root)
    cycle_info_csv = Path(args.cycle_info)

    # cycle -> month
    cycle_month = load_cycle_month_map(cycle_info_csv)
    
    def cycle_month_value(cyc: int) -> int:
        m = cycle_month.get(int(cyc), None)
        if m is None:
            return 1
        return int(m)

    # Topography interpolator (optional)
    topo_interp = None
    if args.topo is not None:
        if args.topo_file is None:
            raise SystemExit("Vous avez spécifié --topo sans --topo-file.")
        topo_interp = _build_topo_interpolator(args.topo_file)
        print(f"[topo] Filtre profondeur actif: topo <= {args.topo:.0f} m (fichier: {args.topo_file})")

    # Grid for binning
    lon_edges, lat_edges, lon_c, lat_c = make_grid(res=args.res)
    ny, nx = lat_c.size, lon_c.size

    weights_line = np.cos(np.deg2rad(lat_c))[:, None]  # (ny,1)

    nbS = args.nbins_strain
    nbZ = args.nbins_vort
    edges_Sn = np.linspace(0.0, args.strain_max, nbS + 1)              # S/f >= 0
    edges_Z  = np.linspace(-args.vort_max, args.vort_max, nbZ + 1)     # zeta/f

    dZ = edges_Z[1] - edges_Z[0]
    dS = edges_Sn[1] - edges_Sn[0]

    Z_centers = 0.5 * (edges_Z[:-1]  + edges_Z[1:])
    S_centers = 0.5 * (edges_Sn[:-1] + edges_Sn[1:])
    ZZ, SS = np.meshgrid(Z_centers, S_centers)  # (nS, nZ)

    #  figure 
    fig, ax = plt.subplots(figsize=(6.2, 5.0), constrained_layout=True)


    base_cmap = plt.get_cmap("twilight")

    months = np.arange(1, 13)
    month_phase = ((months - 2) / 12.0) % 1.0
    month_colors = base_cmap(month_phase)
    # luminance perceptuelle approximative des 12 couleurs mensuelles
    month_luminance = {
        m: 0.2126 * month_colors[m-1][0] + 0.7152 * month_colors[m-1][1] + 0.0722 * month_colors[m-1][2]
        for m in range(1, 13)}
    
    cmap = ListedColormap(month_colors, name="twilight_months_feb_edge_aug_center")
    bounds = np.arange(0.5, 13.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    
    cycles = list(range(args.cycle_start, args.cycle_end + 1))
    
    # dark before
    cycles_sorted = sorted(
        cycles,
        key=lambda c: month_luminance[cycle_month_value(c)]
    )
    print(f"\n=== Overlay contours cycles {args.cycle_start:03d}..{args.cycle_end:03d} ===")
    for c in tqdm(cycles_sorted, desc="cycles"):
        cdir = l3_root / f"cycle_{c:03d}"
        files = sorted(cdir.glob("SWOT_L3_LR_SSH_Expert_*.nc"))
        if not files:
            print(f"[WARN] cycle_{c:03d}: aucun fichier -> skip")
            continue

        hist2d = np.zeros((nbZ, nbS), dtype=float)  # [zeta_bin, S_bin]

        for f in files:
            out = read_vars_from_nc(f, topo_interp=topo_interp, min_depth=args.topo)
            if out is None:
                continue
            LON, LAT, SSH, U, V = out

            Smean, Zmean = strain_vort_binned(
                LON, LAT, U, V,
                lon_edges=lon_edges,
                lat_edges=lat_edges,
                min_count=args.min_count
            )

            f_line = 2 * OMEGA * np.sin(np.deg2rad(lat_c))  # (ny,)
            with np.errstate(invalid="ignore", divide="ignore"):
                Snorm = Smean / f_line[:, None]

            m = np.isfinite(Snorm) & np.isfinite(Zmean)
            if not np.any(m):
                continue

            Svals = Snorm[m]
            Zvals = Zmean[m]

            if args.area_weight:
                iy, ix = np.where(m)
                w = weights_line[iy, 0]
            else:
                w = None

            H, _, _ = np.histogram2d(
                Zvals, Svals,
                bins=(edges_Z, edges_Sn),
                weights=w
            )
            hist2d += H

        total_mass = hist2d.sum() * dZ * dS
        if total_mass <= 0 or not np.isfinite(total_mass):
            print(f"[WARN] cycle_{c:03d}: histogramme vide -> skip")
            continue

        pdf2d = hist2d / total_mass  # joint PDF
        pdf2d[pdf2d == 0] = np.nan

        # compute HDR threshold for chosen enclosed mass
        t = hdr_threshold(pdf2d, dZ=dZ, dS=dS, mass=args.hdr_mass)
        if not np.isfinite(t):
            print(f"[WARN] cycle_{c:03d}: seuil HDR invalide -> skip")
            continue

        month = cycle_month_value(c)
        color = cmap(norm(month))

        ax.contour(
            ZZ, SS,
            pdf2d.T,
            levels=[t],
            colors=[color],
            linewidths=0.8,
            alpha=1,
        )

    x_line = np.linspace(edges_Z[0], edges_Z[-1], 400)
    ax.plot(x_line, x_line, linestyle="--", linewidth=1.0, color="k")
    ax.plot(x_line, -x_line, linestyle="--", linewidth=1.0, color="k")

    ax.set_xlabel(r"$\zeta/f$", fontsize=12)
    ax.set_ylabel(r"$S/f$", fontsize=12)
    ax.set_title("SWOT L3 v2.0.1", fontsize=12)

    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 2)

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

    if args.outfile is not None:
        outpath = Path(args.outfile)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        print(f"Figure enregistrée dans : {outpath}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
