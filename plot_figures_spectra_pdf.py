#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 15:20:24 2025

@author: yan
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter  
from matplotlib.cm import ScalarMappable
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 12})

# ============
#   Tools
# ============

def parse_cycles(tokens):
    if tokens is None or len(tokens) == 0:
        return None  # None => "tous les cycles disponibles" (si besoin)
    sel = set()
    for t in tokens:
        t = str(t).strip()
        if "-" in t:
            a, b = t.split("-", 1)
            a, b = int(a), int(b)
            if a > b:
                a, b = b, a
            sel.update(range(a, b + 1))
        else:
            sel.add(int(t))
    return sorted(sel)


def load_csv_required(csv_path, expected_cols):
    """
    load csv files 
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[WARN] Fichier absent: {csv_path}", file=sys.stderr)
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Échec lecture {csv_path}: {e}", file=sys.stderr)
        return None

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        print(f"[WARN] Colonnes manquantes dans {csv_path}: {missing}", file=sys.stderr)
        return None

    return df


def filter_cycles(df, cycle_list, cycle_col="cycle"):
    if df is None:
        return None, []
    def to_int_safe(x):
        try:
            return int(x)
        except Exception:
            try:
                return int(str(x).strip())
            except Exception:
                return None

    cyc_int = df[cycle_col].apply(to_int_safe)
    df = df.copy()
    df["_cycle_int"] = cyc_int

    if cycle_list is None:  
        kept = df[df["_cycle_int"].notna()]
    else:
        sel = set(int(c) for c in cycle_list)
        kept = df[df["_cycle_int"].isin(sel)]

    found = sorted(set(kept["_cycle_int"].dropna().astype(int).tolist()))
    return kept, found


def build_common_grid_from_pdfs(dfs_by_cycle, npts=400):
    xmin, xmax = np.inf, -np.inf
    for df in dfs_by_cycle:
        if df is None or df.empty:
            continue
        x = df["bin_center"].values
        if not np.all(np.isfinite(x)):
            x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        xmin = min(xmin, np.min(x))
        xmax = max(xmax, np.max(x))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        return None
    return np.linspace(xmin, xmax, npts)


def average_pdf_on_grid(dfs_by_cycle, xgrid):
    Ys = []
    for df in dfs_by_cycle:
        if df is None or df.empty:
            continue
        x = df["bin_center"].values
        y = df["pdf"].values
        m = np.isfinite(x) & np.isfinite(y) & (y >= 0)
        x, y = x[m], y[m]
        if x.size < 3:
            continue
        order = np.argsort(x)
        x, y = x[order], y[order]
        yi = np.interp(xgrid, x, y, left=0.0, right=0.0)
        Ys.append(yi)
    if not Ys:
        return None, 0
    Y = np.vstack(Ys)
    ymean = np.nanmean(Y, axis=0)
    return ymean, Y.shape[0]


def build_common_log_grid_for_spectra(dfs_by_cycle, npts=300, k_col="k_cpkm"):
    lo, hi = -np.inf, np.inf
    found_any = False
    for df in dfs_by_cycle:
        if df is None or df.empty:
            continue
        k = df[k_col].values
        m = np.isfinite(k) & (k > 0)
        if not np.any(m):
            continue
        kk = k[m]
        found_any = True
        lo = max(lo, np.min(kk))
        hi = min(hi, np.max(kk))
    if (not found_any) or (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi <= lo):
        return None
    return np.logspace(np.log10(lo), np.log10(hi), npts)


def average_spectrum_on_log_grid(dfs_by_cycle, kgrid, k_col="k_cpkm", e_col="E_ssh"):
    """
    Interpolate each spectra on the new grid 
    """
    Ys = []
    logk = np.log10(kgrid)
    for df in dfs_by_cycle:
        if df is None or df.empty:
            continue
        k = df[k_col].values
        e = df[e_col].values
        m = np.isfinite(k) & np.isfinite(e) & (k > 0) & (e > 0)
        if not np.any(m):
            continue
        k, e = k[m], e[m]
        order = np.argsort(k)
        k, e = k[order], e[order]
        lk, le = np.log10(k), np.log10(e)
        le_i = np.interp(logk, lk, le, left=np.nan, right=np.nan)
        e_i = 10.0 ** le_i
        Ys.append(e_i)
    if not Ys:
        return None, 0
    Y = np.vstack(Ys)
    Emean = np.nanmean(Y, axis=0)
    return Emean, Y.shape[0]


def split_by_cycle(df, cycle_col="cycle"):
    if df is None or df.empty:
        return {}
    def to_int_safe(x):
        try:
            return int(x)
        except Exception:
            try:
                return int(str(x).strip())
            except Exception:
                return None
    d = {}
    for cyc, dfc in df.groupby(cycle_col):
        ci = to_int_safe(cyc)
        if ci is None:
            continue
        d[ci] = dfc.copy()
    return d


def load_cycle_month_map(csv_path):
    """
    Read compare_ke_l3_duacs_l4_swot_like.csv (cycle, date_min, date_max) to obtain time information
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[WARN] cycle_info CSV not found: {csv_path}", file=sys.stderr)
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Failed to read cycle_info CSV {csv_path}: {e}", file=sys.stderr)
        return None

    required = ["cycle", "date_min", "date_max"]
    if any(c not in df.columns for c in required):
        print(f"[WARN] Missing columns in {csv_path}, expected {required}", file=sys.stderr)
        return None

    def to_int_safe(x):
        try:
            return int(x)
        except Exception:
            try:
                return int(str(x).strip())
            except Exception:
                return None

    df = df.copy()
    df["cycle_int"] = df["cycle"].apply(to_int_safe)
    dmin = pd.to_datetime(df["date_min"], dayfirst=True, errors="coerce")
    dmax = pd.to_datetime(df["date_max"], dayfirst=True, errors="coerce")
    
    mid = dmin + (dmax - dmin) / 2.0

    mid = mid.where(mid.notna(), dmin)
    mid = mid.where(mid.notna(), dmax)
    
    months = mid.dt.month
    mapping = {}
    for ci, m in zip(df["cycle_int"], months):
        if pd.notna(ci) and pd.notna(m):
            mapping[int(ci)] = int(m)
    if not mapping:
        print(f"[WARN] No valid cycle/month mapping built from {csv_path}", file=sys.stderr)
        return None
    print("[DEBUG] cycle_info rows:", len(df))
    print("[DEBUG] NaT dmin:", dmin.isna().sum(), "NaT dmax:", dmax.isna().sum())
    print("[DEBUG] mapping size:", len(mapping))

    return mapping

def build_twilight_month_cmap():
    base_cmap = plt.get_cmap("twilight")
    months = np.arange(1, 13)
    month_phase = ((months - 2) / 12.0) % 1.0
    month_colors = base_cmap(month_phase)

    month_luminance = {
        m: 0.2126 * month_colors[m-1][0]
         + 0.7152 * month_colors[m-1][1]
         + 0.0722 * month_colors[m-1][2]
        for m in range(1, 13)
    }

    cmap = ListedColormap(month_colors, name="twilight_months_feb_edge_aug_center")
    bounds = np.arange(0.5, 13.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)

    return cmap, norm, bounds, month_luminance

# ==========================
#   Products
# ==========================

def load_product_tables(prod_name, prod_dir):
    prod_dir = Path(prod_dir)
    tabs = {
        "strain": load_csv_required(prod_dir / "pdf_strain.csv", ["cycle", "bin_center", "pdf"]),
        "vort":   load_csv_required(prod_dir / "pdf_vort_norm.csv", ["cycle", "bin_center", "pdf"]),
        "spec":   load_csv_required(prod_dir / "ssh_spectra.csv", ["cycle", "k_cpkm", "E_ssh"]),
    }
    return tabs


# ================
#   Plot functions
# ================

def plot_pdfs_compare(products, cycles, cycle_month_map=None, quantity="strain", title=None):

    assert quantity in ("strain", "vort")

    # ordre des panels (L3, L4, DUACS, L3 v3 en 4ᵉ)
    product_order = ["L3 v2.0.1","L3 v3",  "DUACS"]#"L4 MIOST",

    cmaps = {
        "L3 v2.0.1": plt.get_cmap("viridis"),
        "L3 v3": plt.get_cmap("magma"),
        "DUACS": plt.get_cmap("cividis"),
    } #"L4 MIOST": plt.get_cmap("plasma"),
    # styles pour les courbes moyennes (toujours en noir)
    mean_styles = {
        "L3 v2.0.1": "-",
        "L3 v3": "-.",
        "DUACS": ":",
    } #        "L4 MIOST": "--",

    # mois "extrêmes" (hiver + été) à mettre au premier plan
    extreme_months = {1, 2, 12, 6, 7, 8}

    # 4 panels au lieu de 3
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.98, wspace=0.25)


    pre = {}
    for name in product_order:
        df = products.get(name, None)
        if df is None:
            continue
        df_filt, found = filter_cycles(df, cycles, cycle_col="cycle")
        if not found:
            continue
        per_cycle = split_by_cycle(df_filt, cycle_col="cycle")
        dfs = [per_cycle.get(c, None) for c in found]
        xgrid = build_common_grid_from_pdfs(dfs, npts=400)
        mean_x = mean_y = None
        if xgrid is not None:
            ymean, neff = average_pdf_on_grid(dfs, xgrid)
            if ymean is not None and neff > 0:
                mean_x, mean_y = xgrid, ymean
        pre[name] = {
            "found": found,
            "per_cycle": per_cycle,
            "mean_x": mean_x,
            "mean_y": mean_y,
        }

    # --- Panels
    for ax, focus_name in zip(axes, product_order):
        if focus_name not in pre:
            continue

        focus_info = pre[focus_name]
        cmap = cmaps[focus_name]

        # 1) Courbes de cycles pour le produit focus (non extrêmes d'abord)
        for pass_extreme in (False, True):
            for c in focus_info["found"]:
                d = focus_info["per_cycle"].get(c, None)
                if d is None or d.empty:
                    continue
                x = d["bin_center"].values
                y = d["pdf"].values
                m = np.isfinite(x) & np.isfinite(y) & (y >= 0)
                if not np.any(m):
                    continue

                month = None
                if cycle_month_map is not None:
                    month = cycle_month_map.get(int(c))

                is_extreme = (month in extreme_months)
                if pass_extreme != is_extreme:
                    continue

                if month is None:
                    w = 0.5
                else:
                    w = MONTH_TO_SEASON_WEIGHT.get(month, 0.5)
                color = cmap(w)

                lw = 1.3 if is_extreme else 1.0
                z = 3 if is_extreme else 1
                ax.plot(x[m], y[m], lw=lw, alpha=0.8, color=color, zorder=z)

        # 2) Courbes moyennes des autres produits (en noir)
        legend_handles = []
        legend_handles.append(
            Line2D([0], [0], color=cmap(0.5), lw=1.5, label=f"{focus_name} cycles")
        )
        for other in product_order:
            if other == focus_name:
                continue
            info = pre.get(other, None)
            if info is None or info["mean_x"] is None:
                continue
            ax.plot(
                info["mean_x"],
                info["mean_y"],
                color="black",
                lw=2.0,
                linestyle=mean_styles.get(other, "-"),
                zorder=2,
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="black",
                    lw=2.0,
                    linestyle=mean_styles.get(other, "-"),
                    label=f"{other} mean",
                )
            )

        # Légende seulement pour STRAIN (on enlève pour VORT)
        if quantity != "vort":
            ax.legend(handles=legend_handles, fontsize=12, loc="best")

        # Axes
        if quantity == "strain":
            ax.set_xlabel("S", fontsize=12)
            ax.set_xlim(0, 1.3e-4)
        else:
            ax.set_xlabel(r"$\zeta/f$", fontsize=12)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=12)

    # y-label uniquement sur le premier panel
    axes[0].set_ylabel("PDF", fontsize=12)

    # Notation scientifique pour STRAIN (x et y, sur tous les panels)
    if quantity == "strain":
        for ax in axes:
            sf = ScalarFormatter(useMathText=True)
            sf.set_powerlimits((-2, 2))
            ax.xaxis.set_major_formatter(sf)
            ax.yaxis.set_major_formatter(sf)
            ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, 2))

    # Limites spécifiques pour VORT
    if quantity == "vort":
        # panel 0 : x in [-2, 2], y max = 7
        axes[0].set_xlim(-1, 1)
        ylo, yhi = axes[0].get_ylim()
        axes[0].set_ylim(ylo, 7)

        # panels 1,2,3 : x in [-0.5, 0.5]
        if len(axes) > 1:
            axes[1].set_xlim(-0.5, 0.5)
        if len(axes) > 2:
            axes[2].set_xlim(-0.5, 0.5)
        if len(axes) > 3:
            axes[3].set_xlim(-0.5, 0.5)

    # --- Colorbars : uniquement pour VORT
    if quantity == "vort":
        fig.subplots_adjust(
            top=0.82,
            bottom=0.12,
            left=0.06,
            right=0.98,
            wspace=0.25,
        )
        cbar_height = 0.01
        cbar_gap = 0.07

        for ax, focus_name in zip(axes, product_order):
            if focus_name not in pre:
                continue

            cmap = cmaps[focus_name]
            norm = Normalize(vmin=0.0, vmax=1.0)
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])

            bbox = ax.get_position()

            cax = fig.add_axes([
                bbox.x0,
                bbox.y1 + cbar_gap,
                bbox.width,
                cbar_height,
            ])

            cbar = fig.colorbar(
                sm,
                cax=cax,
                orientation="horizontal",
            )
            cbar.set_label(f"Month ({focus_name})", fontsize=12)
            cbar.ax.xaxis.set_label_position('top')
            cbar.ax.xaxis.tick_bottom()

            tick_pos = [MONTH_TO_SEASON_WEIGHT[m] for m in range(1, 13)]
            cbar.set_ticks(tick_pos)
            cbar.set_ticklabels([str(m) for m in range(1, 13)])
            cbar.ax.tick_params(labelsize=12)


def plot_spectra_compare(products, cycles, cycle_month_map=None, title=None):

    product_order = ["L3 v2.0.1", "L3 v3", "DUACS"]

    cmap_month, norm_month, bounds_month, month_luminance = build_twilight_month_cmap()
    
    mean_styles = {
        "L3 v2.0.1": "-",
        "L3 v3": "--",
        "DUACS": ":",
    }

    """
    2 panels :
      (b) L3 v3 spectra colorés + moyenne DUACS en noir
      (c) DUACS spectra colorés + moyenne L3 v3 en noir
    
    # --- AVANT: ["L3 v2.0.1", "L3 v3", "DUACS"]
    product_order = ["L3 v2.0.1", "DUACS"]

    cmaps = {
        "L3 v2.0.1": plt.get_cmap("plasma"),
        "DUACS": plt.get_cmap("cividis"),
    }
    mean_styles = {
        "L3 v2.0.1": "--",
        "DUACS": ":",
    }
    """
    extreme_months = {2, 1, 12, 6, 7, 8}

    def add_powerlaw_guides(ax, series_ke, color="0.5", ls="--", lw=1.2, alpha=0.7):
       xmin, xmax = ax.get_xlim()
       ymin, ymax = ax.get_ylim()
       if (xmin <= 0) or (xmax <= 0) or (ymin <= 0) or (ymax <= 0):
           return
   
       def _clip(klo, khi):
           klo = max(klo, xmin)
           khi = min(khi, xmax)
           if khi <= klo:
               return None
           return klo, khi
   
       def _interp_loglog(k, e, k0):
           k = np.asarray(k); e = np.asarray(e)
           m = np.isfinite(k) & np.isfinite(e) & (k > 0) & (e > 0)
           if np.sum(m) < 3:
               return np.nan
           k = k[m]; e = e[m]
           o = np.argsort(k)
           k = k[o]; e = e[o]
           if (k0 < k[0]) or (k0 > k[-1]):
               return np.nan
           return 10 ** np.interp(np.log10(k0), np.log10(k), np.log10(e))
   
       def _anchor_from_data(k0, mode="median"):
           vals = []
           for k, e in series_ke:
               v = _interp_loglog(k, e, k0)
               if np.isfinite(v):
                   vals.append(v)
           if not vals:
               return np.nan
           vals = np.array(vals)
           if mode == "p95":
               return np.nanpercentile(vals, 95)
           if mode == "p05":
               return np.nanpercentile(vals, 5)
           return np.nanmedian(vals)
   
       def _draw(slope, klo, khi, y0, label, text_pos=0.75):
           k = np.logspace(np.log10(klo), np.log10(khi), 200)
           k0 = np.sqrt(klo * khi)
           y = y0 * (k / k0) ** slope
   
           y_max = np.nanmax(y); y_min = np.nanmin(y)
           if np.isfinite(y_max) and (y_max > 0.92 * ymax):
               fac = (0.92 * ymax) / y_max
               y *= fac; y0 *= fac
           if np.isfinite(y_min) and (y_min < 1.10 * ymin):
               fac = (1.10 * ymin) / y_min
               y *= fac; y0 *= fac
   
           ax.loglog(k, y, ls=ls, lw=lw, color=color, alpha=alpha, zorder=0)
   
           kt = klo * (khi / klo) ** text_pos
           yt = y0 * (kt / np.sqrt(klo * khi)) ** slope
           if np.isfinite(yt) and (yt > 0):
               ax.text(kt, yt*2, label, color=color, fontsize=12, ha="left", va="center")
   
       # ----- k^{-11/3} : 1e-2 -> 1e-1 (au-dessus)
       r = _clip(1e-2, 1e-1)
       if r is not None:
           klo, khi = r
           k0 = np.sqrt(klo * khi)
           a = _anchor_from_data(k0, mode="p95")
           if np.isfinite(a):
               _draw(-11.0/3.0, klo, khi, y0=a * 25, label=r"$k^{-11/3}$", text_pos=0.70)
   
       # ----- k^{-5} : 1e-2 -> 1e-1 (en dessous)
       r = _clip(1e-2, 1e-1)
       if r is not None:
           klo, khi = r
           k0 = np.sqrt(klo * khi)
           a = _anchor_from_data(k0, mode="p05")
           if np.isfinite(a):
               _draw(-5.0, klo, khi, y0=a * 0.001, label=r"$k^{-5}$", text_pos=0.70)
    """
       # ----- k^{-2} : 1e-1 -> 0.3
       r = _clip(1e-1, 0.3)
       if r is not None:
           klo, khi = r
           k0 = np.sqrt(klo * khi)
           a = _anchor_from_data(k0, mode="median")
           if np.isfinite(a):
               _draw(-2.0, klo, khi, y0=a * 0.001, label=r"$k^{-2}$", text_pos=0.65)
    """

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    pre = {}
    for name in product_order:
        df = products.get(name, None)
        if df is None:
            continue
        df_filt, found = filter_cycles(df, cycles, cycle_col="cycle")
        if not found:
            continue
        per_cycle = split_by_cycle(df_filt, cycle_col="cycle")
        dfs = [per_cycle.get(c, None) for c in found]
        kgrid = build_common_log_grid_for_spectra(dfs, npts=300, k_col="k_cpkm")
        mean_k = mean_E = None
        if kgrid is not None:
            Emean, neff = average_spectrum_on_log_grid(dfs, kgrid, k_col="k_cpkm", e_col="E_ssh")
            if Emean is not None and neff > 0:
                mean_k, mean_E = kgrid, Emean
        pre[name] = {
            "found": found,
            "per_cycle": per_cycle,
            "mean_k": mean_k,
            "mean_E": mean_E,
        }

    for ax, focus_name in zip(axes, product_order):
        if focus_name not in pre:
            continue

        focus_info = pre[focus_name]
        cmap = cmap_month
        
        # 1) courbes colorées pour le produit focus (non extrêmes puis extrêmes)
        for pass_extreme in (False, True):
            for c in focus_info["found"]:
                d = focus_info["per_cycle"].get(c, None)
                if d is None or d.empty:
                    continue
                k = d["k_cpkm"].values
                e = d["E_ssh"].values
                m = np.isfinite(k) & np.isfinite(e) & (k > 0) & (e > 0)
                if not np.any(m):
                    continue

                month = None
                if cycle_month_map is not None:
                    month = cycle_month_map.get(int(c))
                is_extreme = (month in extreme_months)
                if pass_extreme != is_extreme:
                    continue

                if month is None:
                    month = 1
                color = cmap(norm_month(month))
                
                lw = 1.3 if is_extreme else 1.0
                z = 3 if is_extreme else 1
                ax.loglog(k[m], e[m], lw=lw, alpha=0.8, color=color, zorder=z)

        # 2) moyennes des deux autres produits (noir)
        legend_handles = []
        legend_handles.append(
            Line2D([0], [0], color=cmap(norm_month(1)), lw=1.5, label=f"{focus_name} cycles")
        )
        for other in product_order:
            if other == focus_name:
                continue
            info = pre.get(other, None)
            if info is None or info["mean_k"] is None:
                continue
            ax.loglog(
                info["mean_k"],
                info["mean_E"],
                color="black",
                lw=2.0,
                linestyle=mean_styles.get(other, "-"),
                zorder=10,
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="black",
                    lw=2.0,
                    linestyle=mean_styles.get(other, "-"),
                    label=f"{other} mean",
                )
            )
            # --- Ajout des pentes de référence (k^-5, k^-2, k^-11/3)
        # --- séries (k,E) du panel pour ancrer les guides
        series_ke = []
        for c in focus_info["found"]:
            d = focus_info["per_cycle"].get(c, None)
            if d is None or d.empty:
                continue
            k = d["k_cpkm"].values
            e = d["E_ssh"].values
            m = np.isfinite(k) & np.isfinite(e) & (k > 0) & (e > 0)
            if np.any(m):
                series_ke.append((k[m], e[m]))

        add_powerlaw_guides(ax, series_ke)

        ax.legend(handles=legend_handles, fontsize=12, loc="best")
        ax.set_xlabel("k (cycles/km)", fontsize=12)
        ax.set_xlim(right=1/2,left=1/800)
        ax.set_ylim(1e-11)
        ax.grid(True, which="both", alpha=0.25)
        ax.tick_params(axis="both", labelsize=12)

    axes[0].set_ylabel("E_ssh (m²·cpkm⁻¹)", fontsize=12)

    #fig.subplots_adjust(top=0.85, bottom=0.12)
    cbar_height = 0.03

    for ax, focus_name in zip(axes, product_order):
        if focus_name not in pre:
            continue
        sm = ScalarMappable(cmap=cmap_month, norm=norm_month)
        sm.set_array([])

        cbar = fig.colorbar(
            sm,
            ax=ax,
            orientation="horizontal",
            location="top",
            fraction=cbar_height,
            boundaries=bounds_month,
            ticks=np.arange(1, 13),
            spacing="proportional",
        )
        cbar.set_label(f"Month ({focus_name})", fontsize=12)
        cbar.ax.tick_params(labelsize=12)

# =============
#   Main
# =============

def main():
    ap = argparse.ArgumentParser(
        description="Compare PDFs (strain, ζ/f) and SSH spectra between L3, L4 MIOST, DUACS."
    )
    ap.add_argument("--l3_dir",default="derive_metrics_100", required=True, help="L3 directory with pdf_*.csv and ssh_spectra.csv")
    ap.add_argument("--l4_miost_dir", default="derive_metrics_l4",required=True, help="L4 MIOST directory with same CSVs")
    ap.add_argument("--duacs_dir",default="derive_metrics_l4_duacs", required=True, help="DUACS directory with same CSVs")
    ap.add_argument("--cycles", nargs="+", default=None,
                    help="List/ranges of cycles (e.g. 1 3 5-10). If a single cycle, simple display; if several, per-cycle curves.")
    ap.add_argument("--title_prefix", default="", help="Prefix for figure titles (unused now)")
    ap.add_argument("--savefig", default=None,
                    help="Directory where to save figures (png). If not specified, only display.")
    ap.add_argument("--cycle_info_csv", default="compare_ke_l3_l4.csv",
                    help="CSV with columns cycle,date_min,date_max to derive cycle month.")
    ap.add_argument("--l3_v3_dir", default="derive_metrics_v3",required=False,
                    help="L3 v3 directory with pdf_*.csv and ssh_spectra.csv")

    args = ap.parse_args()

    # Map cycle -> month from compare_ke_l3_duacs_l4_swot_like.csv
    cycle_month_map = load_cycle_month_map(args.cycle_info_csv)

    # Parse cycles
    cycles = parse_cycles(args.cycles)
    if cycles is None:
        title_suffix = "(all available cycles)"
    elif len(cycles) == 1:
        title_suffix = f"(cycle {cycles[0]})"
    else:
        if len(cycles) >= 2:
            title_suffix = f"(cycles {min(cycles)}–{max(cycles)})"
        else:
            title_suffix = "(selected cycles)"

    # Charge tables
    tabs_L3    = load_product_tables("L3", Path(args.l3_dir))
    tabs_MIOST = load_product_tables("L4 MIOST", Path(args.l4_miost_dir))
    tabs_DUACS = load_product_tables("DUACS", Path(args.duacs_dir))

    # L3 v3 (optionnel, mais utilisé pour le 4ᵉ panel)
    if args.l3_v3_dir is not None:
        tabs_L3v3 = load_product_tables("L3 v3", Path(args.l3_v3_dir))
    else:
        tabs_L3v3 = {"strain": None, "vort": None, "spec": None}

    """
    # STRAIN
    # STRAIN
    prod_pdfs_strain = {
        "L3 v2.0.1":      tabs_L3["strain"],
        "L3 v3":   tabs_L3v3["strain"],
        "L4 MIOST": tabs_MIOST["strain"],
        "DUACS":   tabs_DUACS["strain"],
    }
    plot_pdfs_compare(
        prod_pdfs_strain,
        cycles,
        cycle_month_map=cycle_month_map,
        quantity="strain",
        title=None
    )
    if args.savefig:
        Path(args.savefig).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(args.savefig) / f"compare_strain_{title_suffix.replace(' ','_')}.png", dpi=300)

    # ζ/f
    prod_pdfs_vort = {
        "L3 v2.0.1":      tabs_L3["vort"],
        "L3 v3":   tabs_L3v3["vort"],
        "L4 MIOST": tabs_MIOST["vort"],
        "DUACS":   tabs_DUACS["vort"],
    }

    plot_pdfs_compare(
        prod_pdfs_vort,
        cycles,
        cycle_month_map=cycle_month_map,
        quantity="vort",
        title=None
    )
    if args.savefig:
        plt.savefig(Path(args.savefig) / f"compare_vortnorm_{title_suffix.replace(' ','_')}.png", dpi=300)
    """
    # Spectres SSH
    prod_specs = {
        "L3 v2.0.1": tabs_L3["spec"],
        "L3 v3": tabs_L3v3["spec"],
        "DUACS": tabs_DUACS["spec"],
    }
    plot_spectra_compare(
        prod_specs,
        cycles,
        cycle_month_map=cycle_month_map,
        title=None
    )
    if args.savefig:
        plt.savefig(Path(args.savefig) / f"compare_spectra_{title_suffix.replace(' ','_')}.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    main()
