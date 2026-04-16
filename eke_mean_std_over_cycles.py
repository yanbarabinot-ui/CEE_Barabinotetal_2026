#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 09:52:47 2025

@author: yan
"""

"""
EKE mean and std over SWOT cycles.
"""

import argparse
import importlib
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean 

DEFAULT_BASE_OUTDIR = Path("swot_l3_multi_cycles_v3")
DEFAULT_PATTERN = "SWOT_L3_LR_SSH_Expert_*.nc"
DEFAULT_RES = 0.025
DEFAULT_MIN_COUNT = 1

cke = importlib.import_module("compute_med_ke_binned")


def parse_cycles(tokens: Optional[List[str]], available_hint: Optional[Iterable[int]] = None) -> List[int]:
    if tokens is None or tokens == ["all"]:
        if available_hint is None:
            raise ValueError("Vous avez demandé 'all' mais la liste des cycles disponibles est inconnue.")
        return sorted(set(int(c) for c in available_hint))
    sel = set()
    for tok in tokens:
        if re.search(r"^\d+\-\d+$", tok):
            a, b = tok.split("-", 1)
            sel.update(range(int(a), int(b) + 1))
        else:
            sel.add(int(tok))
    return sorted(sel)

def find_cycle_dir(base_outdir: Path, cycle: int) -> Path:
    return base_outdir / f"cycle_{cycle:03d}"


# ---------- EKE per cycle ----------
def compute_cycle_ke_map(cycle_dir: Path, pattern: str, res: float, min_count: int):
    files = sorted(cycle_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Aucun .nc ne correspond à {pattern} dans {cycle_dir}")

    lon_edges, lat_edges, lon_centers, lat_centers = cke.make_grid(
        lon_min=-6, lon_max=36, lat_min=30, lat_max=46, res=res
    )
    Ny, Nx = len(lat_edges) - 1, len(lon_edges) - 1
    sum_grid  = np.zeros((Ny, Nx), dtype=float)
    sum2_grid = np.zeros((Ny, Nx), dtype=float)
    cnt_grid  = np.zeros((Ny, Nx), dtype=float)

    for nc in files:
        out = cke.read_ke_from_nc(nc)
        if out is None:
            continue
        lon, lat, ke = out        # déjà sur le ROI Med, filtré, 1D
        cke.accumulate_bins(lon, lat, ke, lon_edges, lat_edges, sum_grid, sum2_grid, cnt_grid)

    with np.errstate(invalid="ignore", divide="ignore"):
        ke_mean = sum_grid / cnt_grid
        ke_mean[cnt_grid < min_count] = np.nan

    return lon_centers, lat_centers, ke_mean


def aggregate_with_mask(stack: np.ndarray, min_coverage: int):
    """(Ncycles, Ny, Nx) → eke_mean, eke_std, valid_counts, coverage_mask."""
    valid_counts = np.sum(~np.isnan(stack), axis=0)
    coverage_mask = valid_counts >= min_coverage
    masked = stack.copy()
    masked[:, ~coverage_mask] = np.nan
    eke_mean = np.nanmean(masked, axis=0)
    eke_std  = np.nanstd(masked, axis=0, ddof=0)
    eke_mean[~coverage_mask] = np.nan
    eke_std[~coverage_mask]  = np.nan
    return eke_mean, eke_std, valid_counts, coverage_mask

def _raise_gridlines(gl, z=10):
    for coll in (getattr(gl, "xlines", None), getattr(gl, "ylines", None)):
        if coll is None:
            continue
        if hasattr(coll, "set_zorder"):
            coll.set_zorder(z)
        else:
            try:
                for artist in coll:
                    artist.set_zorder(z)
            except TypeError:
                pass

    for labs in (getattr(gl, "xlabel_artists", []), getattr(gl, "ylabel_artists", [])):
        try:
            for lab in labs:
                lab.set_zorder(z + 0.1)
        except TypeError:
            pass

    try:
        gl.zorder = z
    except Exception:
        pass


# ---------- Visualisation ----------

def plot_map(lon_c, lat_c, Z, title: str, out_png: Path,
             vmin=None, vmax=None, cmap_name="magma", cbar_label=""):
    import matplotlib as mpl
    import numpy as np
    import matplotlib.ticker as mticker
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    # 1) Mercator
    proj_map = ccrs.Mercator()
    proj_data = ccrs.PlateCarree()  # données en lon/lat

    fig = plt.figure(figsize=(10, 5), layout="constrained")
    ax = plt.axes(projection=proj_map)
    ax.set_extent([-6, 36, 30, 46], crs=proj_data) #full med

    # 2) map
    land50 = cfeature.NaturalEarthFeature("physical", "land", "50m")
    ax.add_feature(land50, facecolor="lightgray", edgecolor="black",
                   linewidth=0.5, zorder=3)  # pas de clip_on explicite
    ax.coastlines(resolution="50m", color="black", linewidth=0.6, zorder=4)

    # 3) grid
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
    
    # 4) Colormap 
    cmap = plt.colormaps.get_cmap(cmap_name).copy()
    cmap.set_bad(alpha=0.0)

    # 5) Pcolormesh 
    LON, LAT = np.meshgrid(lon_c, lat_c)
    Zm = np.ma.masked_invalid(Z)
    m = ax.pcolormesh(LON, LAT, Zm, shading="auto",
                      vmin=vmin, vmax=vmax, cmap=cmap,
                      transform=proj_data, zorder=2, rasterized=True)

    # 6) colorbar
    cb = fig.colorbar(m, ax=ax, shrink=0.6,extend='max')
    cb.set_label(cbar_label + " [cm² s⁻²]",fontsize=tick_fontsize)
    cb.ax.tick_params(labelsize=tick_fontsize)
    #ax.set_title(title)
    fig.savefig(out_png, dpi=300)
    print(f"Figure enregistrée : {out_png}")


# ---------- main ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Cartes EKE moyenne et écart-type sur un ensemble de cycles SWOT L3 (Méditerranée)"
    )
    ap.add_argument("--base-outdir", default=str(DEFAULT_BASE_OUTDIR),
                    help="Racine contenant les sous-dossiers par cycle : cycle_001/, cycle_002/, ...")
    ap.add_argument("--cycles", nargs="+", required=True,
                    help="Liste de cycles, ex : 1-10 12 15-17")
    ap.add_argument("--pattern", default=DEFAULT_PATTERN,
                    help="Glob pattern des fichiers .nc (comme dans compute_med_ke_binned.py)")
    ap.add_argument("--res", type=float, default=DEFAULT_RES,
                    help="Résolution des bins en degrés")
    ap.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT,
                    help="Minimum d'observations par bin à conserver")
    ap.add_argument("--min-coverage", type=int, default=None,
                    help="Nb minimal de cycles valides par pixel (défaut : intersection stricte)")
    ap.add_argument("--outdir", default=None,
                    help="Répertoire de sortie pour figures/npz (défaut : base_outdir)")
    ap.add_argument("--vmax-mean", type=float, default=None,
                    help="Coupe supérieure (échelle couleur) pour la carte de moyenne")
    ap.add_argument("--vmax-std", type=float, default=None,
                    help="Coupe supérieure (échelle couleur) pour la carte d'écart-type")
    ap.add_argument("--no-npz", action="store_true",
                    help="Ne pas écrire le .npz agrégé (seulement les figures)")

    args = ap.parse_args()

    base_outdir = Path(args.base_outdir)
    out_root = Path(args.outdir) if args.outdir is not None else base_outdir
    out_root.mkdir(parents=True, exist_ok=True)

    # Cycles
    cycles = parse_cycles(args.cycles)
    spec = "_".join(args.cycles)

    lon_c = lat_c = None
    per_cycle = []
    used_cycles = []

    for c in cycles:
        cyc_dir = find_cycle_dir(base_outdir, c)
        if not cyc_dir.exists():
            print(f"[cycle {c:03d}] Dossier absent : {cyc_dir} — skip.")
            continue
        try:
            lc, lt, ke = compute_cycle_ke_map(cyc_dir, args.pattern, args.res, args.min_count)
        except Exception as e:
            print(f"[cycle {c:03d}] Impossible de calculer KE : {e}")
            continue

        if lon_c is None:
            lon_c, lat_c = lc, lt
        else:
            same = (lc.shape == lon_c.shape) and (lt.shape == lat_c.shape) \
                   and np.allclose(lc, lon_c) and np.allclose(lt, lat_c)
            if not same:
                print(f"[cycle {c:03d}] Grille incompatible — skip.")
                continue

        per_cycle.append(ke.astype(float))
        used_cycles.append(c)
        print(f"[cycle {c:03d}] OK : carte KE empilée.")

    if not per_cycle:
        raise SystemExit("Aucun cycle utilisable. Vérifiez le pattern, la résolution et la présence des .nc.")

    stack = np.stack(per_cycle, axis=0)  # (Ncycles, Ny, Nx)

    # mask
    if args.min_coverage is None:
        min_cov = stack.shape[0]
        coverage_mode = f"intersection_all={min_cov}"
    else:
        min_cov = int(args.min_coverage)
        coverage_mode = f"min_coverage={min_cov}"

    eke_mean, eke_std, valid_counts, coverage_mask = aggregate_with_mask(stack, min_cov)

    # save (optional)
    if not args.no_npz:
        out_npz = out_root / f"eke_mean_std_cycles_{spec}.npz"
        np.savez_compressed(
            out_npz,
            lon_centers=lon_c,
            lat_centers=lat_c,
            eke_mean=eke_mean,
            eke_std=eke_std,
            valid_counts=valid_counts,
            cycles=np.array(used_cycles, dtype=int),
            coverage_mode=coverage_mode,
            res=args.res,
            min_count=args.min_count,
        )
        print(f"Sauvegardé : {out_npz}")

    # Figures
    mean_png = out_root / f"v3_eke_mean_cycles_{spec}.png"
    std_png  = out_root / f"Med_eke_std_cycles_{spec}.png"

    vmax_mean = 1200#args.vmax_mean if args.vmax_mean is not None else np.nanpercentile(eke_mean, 98)
    vmax_std  = args.vmax_std  if args.vmax_std  is not None else np.nanpercentile(eke_std,  98)

    title_mean = f"SWOT L3 - EKE mean for cycles {min(used_cycles)}-{max(used_cycles)}"
    title_std = f"EKE std for cycles {min(used_cycles)}-{max(used_cycles)}"

    plot_map(lon_c, lat_c, 10000*eke_mean,
             title=title_mean,
             out_png=mean_png, vmin=0.0, vmax=vmax_mean, cmap_name='viridis', cbar_label='EKE')
    plot_map(lon_c, lat_c, 10000*eke_std,
             title=title_std, 
             out_png=std_png, vmin=0.0, vmax=10000*vmax_std, cmap_name=cmocean.cm.amp, cbar_label='EKE std')

    print("Terminé.")


