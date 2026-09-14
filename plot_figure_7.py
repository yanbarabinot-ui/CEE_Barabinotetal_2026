#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 16:37:46 2026

@author: yan
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})


# ============================================================
# Helpers
# ============================================================

def load_csv_required(csv_path, expected_cols):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[WARN] Missing file: {csv_path}", file=sys.stderr)
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Failed reading {csv_path}: {e}", file=sys.stderr)
        return None

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        print(f"[WARN] Missing columns in {csv_path}: {missing}", file=sys.stderr)
        return None
    return df


def to_int_safe(x):
    try:
        return int(x)
    except Exception:
        try:
            return int(str(x).strip())
        except Exception:
            return None


def split_by_cycle(df, cycle_col="cycle"):
    if df is None or df.empty:
        return {}
    out = {}
    for cyc, dfc in df.groupby(cycle_col):
        ci = to_int_safe(cyc)
        if ci is None:
            continue
        out[ci] = dfc.copy()
    return out


def load_cycle_time_info(csv_path):
    """
    Lit compare_ke_l3_l4.csv et retourne un DataFrame avec :
    - cycle
    - date_min
    - date_max
    - date_median
    - month_period : mois du jour médian
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[WARN] cycle_info CSV not found: {csv_path}", file=sys.stderr)
        return None

    try:
        df = pd.read_csv(csv_path, parse_dates=["date_min", "date_max"])
    except Exception as e:
        print(f"[WARN] Failed to read {csv_path}: {e}", file=sys.stderr)
        return None

    required = ["cycle", "date_min", "date_max"]
    if any(c not in df.columns for c in required):
        print(f"[WARN] Missing columns in {csv_path}, expected {required}", file=sys.stderr)
        return None

    df = df.copy()
    df["cycle"] = df["cycle"].apply(to_int_safe)
    df = df[df["cycle"].notna()].copy()
    df["cycle"] = df["cycle"].astype(int)

    df["date_median"] = df["date_min"] + (df["date_max"] - df["date_min"]) / 2.0
    df["month_period"] = df["date_median"].dt.to_period("M").dt.to_timestamp()

    df = df[df["date_median"].notna()].copy()

    if df.empty:
        print(f"[WARN] No valid cycle/date information in {csv_path}", file=sys.stderr)
        return None

    return df[["cycle", "date_min", "date_max", "date_median", "month_period"]].drop_duplicates("cycle")

def load_l3v3_spectra(spec_csv):
    df = load_csv_required(spec_csv, ["cycle", "k_cpkm", "E_ssh"])
    if df is None:
        raise SystemExit(f"Cannot read spectra CSV: {spec_csv}")

    df = df.copy()
    df["cycle_int"] = df["cycle"].apply(to_int_safe)
    df = df[df["cycle_int"].notna()].copy()
    df["cycle_int"] = df["cycle_int"].astype(int)

    m = (
        np.isfinite(df["k_cpkm"].values)
        & np.isfinite(df["E_ssh"].values)
        & (df["k_cpkm"].values > 0)
        & (df["E_ssh"].values > 0)
    )
    df = df.loc[m].copy()
    return df


def interp_spectrum_on_kgrid(k, e, kgrid):
    """
    Interpolation log-log sur kgrid.
    Retourne E(kgrid), NaN hors support.
    """
    k = np.asarray(k, dtype=float)
    e = np.asarray(e, dtype=float)
    m = np.isfinite(k) & np.isfinite(e) & (k > 0) & (e > 0)
    if np.sum(m) < 3:
        return None

    k = k[m]
    e = e[m]
    o = np.argsort(k)
    k = k[o]
    e = e[o]

    logk = np.log10(k)
    loge = np.log10(e)
    logkg = np.log10(kgrid)

    eg = np.full_like(kgrid, np.nan, dtype=float)
    inside = (kgrid >= k[0]) & (kgrid <= k[-1])
    if np.any(inside):
        eg[inside] = 10.0 ** np.interp(logkg[inside], logk, loge)
    return eg


def common_kgrid_from_cycles(per_cycle, npts=800):
    """
    Grille commune log-spacée basée sur l'intersection des domaines spectraux.
    """
    lo = -np.inf
    hi = np.inf
    found = False

    for _, dfc in per_cycle.items():
        k = dfc["k_cpkm"].values
        m = np.isfinite(k) & (k > 0)
        if not np.any(m):
            continue
        kk = k[m]
        found = True
        lo = max(lo, np.min(kk))
        hi = min(hi, np.max(kk))

    if (not found) or (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi <= lo):
        return None

    return np.logspace(np.log10(lo), np.log10(hi), npts)


def integrate_on_interval(k, e, kmin, kmax):
    k = np.asarray(k, dtype=float)
    e = np.asarray(e, dtype=float)

    m = np.isfinite(k) & np.isfinite(e) & (k > 0) & (e >= 0)
    if np.sum(m) < 2:
        return np.nan

    k = k[m]
    e = e[m]
    o = np.argsort(k)
    k = k[o]
    e = e[o]

    kmin_eff = max(kmin, k[0])
    kmax_eff = min(kmax, k[-1])

    if kmax_eff <= kmin_eff:
        return np.nan

    kin = (k > kmin_eff) & (k < kmax_eff)
    kk = np.concatenate(([kmin_eff], k[kin], [kmax_eff]))
    ee = np.interp(kk, k, e)

    return np.trapz(ee, kk)


def format_month_year_fr(dt):
    mois = {
        1: "Jan",
        2: "Fév",
        3: "Mars",
        4: "Avr",
        5: "Mai",
        6: "Juin",
        7: "Juil",
        8: "Août",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Déc",
    }
    return f"{mois[dt.month]} {dt.year}"

def integrate_full_range(k, e):
    """
    Intègre e(k) sur tout le domaine disponible.
    """
    k = np.asarray(k, dtype=float)
    e = np.asarray(e, dtype=float)

    m = np.isfinite(k) & np.isfinite(e) & (k > 0) & (e >= 0)
    if np.sum(m) < 2:
        return np.nan

    k = k[m]
    e = e[m]
    o = np.argsort(k)
    k = k[o]
    e = e[o]

    return np.trapz(e, k)


def compute_monthly_band_fractions(df_spec, df_time, bands_km, gf2):
    per_cycle = split_by_cycle(df_spec, cycle_col="cycle_int")
    if not per_cycle:
        return None

    kmin_all = np.inf
    kmax_all = -np.inf
    
    for _, dfc in per_cycle.items():
        k = dfc["k_cpkm"].values
        m = np.isfinite(k) & (k > 0)
        if np.any(m):
            kmin_all = min(kmin_all, np.nanmin(k[m]))
            kmax_all = max(kmax_all, np.nanmax(k[m]))
    
    if (not np.isfinite(kmin_all)) or (not np.isfinite(kmax_all)) or (kmax_all <= kmin_all):
        return None
    
    kgrid = np.logspace(np.log10(kmin_all), np.log10(kmax_all), 800)

    cycle_records = []

    for cyc, dfc in per_cycle.items():
        row_time = df_time.loc[df_time["cycle"] == int(cyc)]
        if row_time.empty:
            continue

        date_median = pd.Timestamp(row_time["date_median"].iloc[0])
        month_period = pd.Timestamp(row_time["month_period"].iloc[0])

        k = dfc["k_cpkm"].values
        e = dfc["E_ssh"].values
        eg = interp_spectrum_on_kgrid(k, e, kgrid)
        if eg is None:
            continue

        print(cyc, np.min(k), np.max(k))
        e_eke = 0.5 * gf2 * (2.0 * np.pi * kgrid / 1000.0) ** 2 * eg

        rec = {
            "cycle": int(cyc),
            "date_mid": date_median,
            "month_period": month_period,
        }

        band_values = {}
        for Lmax, Lmin in bands_km:
            k1 = 1.0 / float(Lmax)
            k2 = 1.0 / float(Lmin)
            band_val = integrate_on_interval(kgrid, e_eke, k1, k2)
            band_values[(Lmax, Lmin)] = band_val
            rec[f"band_{Lmax}_{Lmin}km"] = band_val

        eke_total = integrate_full_range(kgrid, e_eke)
        rec["eke_total"] = eke_total

        for Lmax, Lmin in bands_km:
            band_val = band_values[(Lmax, Lmin)]
            rec[f"eke_{Lmax}_{Lmin}km"] = (
                band_val * 1e4
                if np.isfinite(band_val)
                else np.nan
            )

        cycle_records.append(rec)

    if not cycle_records:
        return None

    df_cycles = pd.DataFrame(cycle_records)
    eke_cols = [f"eke_{a}_{b}km" for a, b in bands_km]

    monthly = (
        df_cycles.groupby("month_period")[eke_cols]
        .mean()
        .reset_index()
        .sort_values("month_period")
    )

    return monthly

def fill_month_for_plot_only(monthly, month, cols):
    """
    Remplit un mois par interpolation linéaire temporelle.
    Uniquement pour le plotting, ne modifie pas les données originales.
    """
    monthly_plot = monthly.copy()
    monthly_plot = monthly_plot.sort_values("month_period").reset_index(drop=True)

    month = pd.Timestamp(month)

    # S'assure que le mois existe dans la série
    if month not in monthly_plot["month_period"].values:
        new_row = {"month_period": month}
        for col in cols:
            new_row[col] = np.nan
        monthly_plot = pd.concat(
            [monthly_plot, pd.DataFrame([new_row])],
            ignore_index=True
        )

    monthly_plot = monthly_plot.sort_values("month_period").reset_index(drop=True)

    # Force le mois cible à NaN puis interpole
    for col in cols:
        monthly_plot.loc[monthly_plot["month_period"] == month, col] = np.nan
        monthly_plot[col] = monthly_plot[col].interpolate(
            method="linear",
            limit_direction="both"
        )

    return monthly_plot


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description=(
            "SWOT L3 v3: monthly averaged band-limited variance "
            "using (g/f)^2 k^2 E_ssh(k) from ssh_spectra.csv"
        )
    )
    ap.add_argument(
        "--l3_v3_dir",
        required=True,
        help="Directory containing ssh_spectra.csv for SWOT L3 v3"
    )
    ap.add_argument(
        "--cycle_info_csv",
        default="compare_ke_l3_l4.csv",
        help="CSV with columns cycle,date_min,date_max"
    )
    ap.add_argument(
        "--outfile",
        default=None,
        help="Output figure path (png/pdf). If omitted, only display."
    )
    ap.add_argument(
        "--duacs_dir",
        required=True,
        help="Directory containing ssh_spectra.csv for DUACS"
    )
    args = ap.parse_args()

    # --------------------------------------------------------
    # Constants for Mediterranean-mean geostrophic conversion
    # --------------------------------------------------------
    g = 9.81  # m s^-2
    omega = 7.2921159e-5  # s^-1
    phi_deg = 37.0  # representative Mediterranean mean latitude
    f_mean = 2.0 * omega * np.sin(np.deg2rad(phi_deg))
    gf2 = (g / f_mean) ** 2

    #print(f"Using g = {g:.3f} m s^-2")
    #print(f"Using mean Mediterranean latitude = {phi_deg:.1f}°N")
    #print(f"Using f_mean = {f_mean:.6e} s^-1")
    #print(f"Using (g/f)^2 = {gf2:.6e}")

    # --------------------------------------------------------
    # Inputs
    # --------------------------------------------------------
    spec_csv = Path(args.l3_v3_dir) / "ssh_spectra.csv"
    df_spec = load_l3v3_spectra(spec_csv)
    # Cycle 4 SWOT is incomplete, keep only cycle 3 for September 2023
    df_spec = df_spec[df_spec["cycle_int"] != 4].copy()
    df_time = load_cycle_time_info(args.cycle_info_csv)

    if df_time is None:
        raise SystemExit("No valid cycle time information available.")

    # --------------------------------------------------------
    # Common spectral grid
    # --------------------------------------------------------

    bands_km = [
        (500, 200),
        (200, 100),
        (100, 60),
        (60, 20),
    ]

    # --------------------------------------------------------
    # Compute per-cycle band variances using (g/f)^2 k^2 E_ssh(k)
    # --------------------------------------------------------
    eke_cols = [f"eke_{a}_{b}km" for a, b in bands_km]

    # SWOT L3 v3
    monthly = compute_monthly_band_fractions(df_spec, df_time, bands_km, gf2)
    if monthly is None:
        raise SystemExit("No SWOT L3 v3 monthly fractions could be computed.")

    print("\nMonths present in SWOT monthly:")
    print(monthly["month_period"].dt.strftime("%Y-%m").tolist())
    
    print("\nCycle/date mapping:")
    print(df_time[["cycle", "date_min", "date_max", "date_median", "month_period"]].to_string(index=False))    
    #print("\nMonthly sum of SWOT fractions:")
    #monthly["frac_sum"] = monthly[eke_cols].sum(axis=1)
    #print(monthly[["month_period", "frac_sum"]])

    # DUACS
    duacs_spec_csv = Path(args.duacs_dir) / "ssh_spectra.csv"
    df_duacs_spec = load_l3v3_spectra(duacs_spec_csv)

    monthly_duacs = compute_monthly_band_fractions(df_duacs_spec, df_time, bands_km, gf2)
    if monthly_duacs is None:
        print("[WARN] No DUACS monthly fractions could be computed.", file=sys.stderr)

# --------------------------------------------------------
# Fill selected problematic months by linear interpolation
# for plotting only
# --------------------------------------------------------
    monthly_plot = fill_month_for_plot_only(
        monthly,
        "2023-09-01",
        eke_cols
    )
    
    if monthly_duacs is not None:
        monthly_duacs_plot = monthly_duacs.copy()
        monthly_duacs_plot = fill_month_for_plot_only(
            monthly_duacs_plot,
            "2024-08-01",
            eke_cols
        )
        monthly_duacs_plot = fill_month_for_plot_only(
            monthly_duacs_plot,
            "2025-08-01",
            eke_cols
        )
    else:
        monthly_duacs_plot = None
    # --------------------------------------------------------
    # Colorblind-friendly palette (Okabe-Ito)
    # --------------------------------------------------------
    color_map = {
        (500, 200): "#0072B2",  # blue
        (200, 100): "#E69F00",  # orange
        (100, 60):  "#009E73",  # green
        (60, 20):   "#CC79A7",  # purple/magenta
    }

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------


    fig, axes = plt.subplots(
        2, 1,
        figsize=(10, 7),
        sharex=True,
        sharey=True,
        constrained_layout=True
    )

    panel_bands = [
        [(500, 200), (200, 100)],
        [(100, 60), (60, 20)],
    ]

    panel_titles = [
        "Large scales",
        "Small scales",
    ]

    for ax, bands_panel, title in zip(axes, panel_bands, panel_titles):

        for a, b in bands_panel:
            col = f"eke_{a}_{b}km"

            ax.plot(
                monthly_plot["month_period"],
                monthly_plot[col],
                marker="o",
                lw=2.2,
                color=color_map[(a, b)],
                label=f"SWOT {a}–{b} km"
            )

            if monthly_duacs is not None:
                ax.plot(
                    monthly_duacs_plot["month_period"],
                    monthly_duacs_plot[col],
                    marker="o",
                    lw=2.0,
                    linestyle="--",
                    color=color_map[(a, b)],
                    label=f"DUACS {a}–{b} km"
                )

        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, loc="upper right", frameon=True)

    axes[0].set_ylabel(r"$EKE(k_1,k_2)$ [cm$^2$ s$^{-2}$]")
    axes[1].set_ylabel(r"$EKE(k_1,k_2)$ [cm$^2$ s$^{-2}$]")

    # X ticks: same logic as the EKE/skewness code
    axes[1].xaxis.set_major_locator(MonthLocator(interval=1))
    axes[1].xaxis.set_major_formatter(DateFormatter("%b %Y"))
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")
    axes[1].set_xlabel("")

    # Same vertical scale for both panels
    ymax_vals = []

    for a, b in bands_km:
        col = f"eke_{a}_{b}km"
        ymax_vals.append(np.nanmax(monthly_plot[col].values))

        if monthly_duacs is not None:
            ymax_vals.append(np.nanmax(monthly_duacs_plot[col].values))

    ymax = np.nanmax(ymax_vals)
    axes[0].set_ylim(0, 1.05 * ymax)

    axes[0].set_title("")
    axes[1].set_title("")

    # Save / show
    if args.outfile is not None:
        outpath = Path(args.outfile)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        print(f"\nFigure saved to: {outpath}")

    plt.show()


if __name__ == "__main__":
    main()