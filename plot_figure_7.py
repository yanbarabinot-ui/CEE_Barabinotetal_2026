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
    - date_mid
    - month_period (début du mois de la date médiane)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[WARN] cycle_info CSV not found: {csv_path}", file=sys.stderr)
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Failed to read {csv_path}: {e}", file=sys.stderr)
        return None

    required = ["cycle", "date_min", "date_max"]
    if any(c not in df.columns for c in required):
        print(f"[WARN] Missing columns in {csv_path}, expected {required}", file=sys.stderr)
        return None

    df = df.copy()
    df["cycle"] = df["cycle"].apply(to_int_safe)

    dmin = pd.to_datetime(df["date_min"], errors="coerce")
    dmax = pd.to_datetime(df["date_max"], errors="coerce")

    mid = dmin + (dmax - dmin) / 2.0
    mid = mid.where(mid.notna(), dmin)
    mid = mid.where(mid.notna(), dmax)

    df["date_min"] = dmin
    df["date_max"] = dmax
    df["date_mid"] = mid
    df["month_period"] = df["date_mid"].dt.to_period("M").dt.to_timestamp()

    df = df[df["cycle"].notna() & df["date_mid"].notna()].copy()
    df["cycle"] = df["cycle"].astype(int)

    if df.empty:
        print(f"[WARN] No valid cycle/date information in {csv_path}", file=sys.stderr)
        return None

    return df[["cycle", "date_min", "date_max", "date_mid", "month_period"]].drop_duplicates("cycle")


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
    """
    Intégration avec interpolation aux bornes exactes.
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

    if (kmin < k[0]) or (kmax > k[-1]) or (kmax <= kmin):
        return np.nan

    kin = (k > kmin) & (k < kmax)
    kk = np.concatenate(([kmin], k[kin], [kmax]))
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
    args = ap.parse_args()

    # --------------------------------------------------------
    # Constants for Mediterranean-mean geostrophic conversion
    # --------------------------------------------------------
    g = 9.81  # m s^-2
    omega = 7.2921159e-5  # s^-1
    phi_deg = 37.0  # representative Mediterranean mean latitude
    f_mean = 2.0 * omega * np.sin(np.deg2rad(phi_deg))
    gf2 = (g / f_mean) ** 2

    print(f"Using g = {g:.3f} m s^-2")
    print(f"Using mean Mediterranean latitude = {phi_deg:.1f}°N")
    print(f"Using f_mean = {f_mean:.6e} s^-1")
    print(f"Using (g/f)^2 = {gf2:.6e}")

    # --------------------------------------------------------
    # Inputs
    # --------------------------------------------------------
    spec_csv = Path(args.l3_v3_dir) / "ssh_spectra.csv"
    df_spec = load_l3v3_spectra(spec_csv)
    df_time = load_cycle_time_info(args.cycle_info_csv)

    if df_time is None:
        raise SystemExit("No valid cycle time information available.")

    per_cycle = split_by_cycle(df_spec, cycle_col="cycle_int")
    if not per_cycle:
        raise SystemExit("No valid spectra found by cycle.")

    # --------------------------------------------------------
    # Common spectral grid
    # --------------------------------------------------------
    kgrid = common_kgrid_from_cycles(per_cycle, npts=800)
    if kgrid is None:
        raise SystemExit("Could not build a common spectral grid.")

    bands_km = [
        (500, 200),
        (200, 100),
        (100, 60),
        (60, 20),
    ]

    # --------------------------------------------------------
    # Compute per-cycle band variances using (g/f)^2 k^2 E_ssh(k)
    # --------------------------------------------------------
    cycle_records = []

    for cyc, dfc in per_cycle.items():
        row_time = df_time.loc[df_time["cycle"] == int(cyc)]
        if row_time.empty:
            continue

        date_mid = pd.Timestamp(row_time["date_mid"].iloc[0])
        month_period = pd.Timestamp(row_time["month_period"].iloc[0])

        k = dfc["k_cpkm"].values
        e = dfc["E_ssh"].values
        eg = interp_spectrum_on_kgrid(k, e, kgrid)
        if eg is None:
            continue

        # Geostrophic EKE spectral proxy with mean Mediterranean (g/f)^2 factor
        e_eke = gf2 * (kgrid ** 2) * eg

        rec = {
            "cycle": int(cyc),
            "date_mid": date_mid,
            "month_period": month_period,
        }

        # Absolute variances by band
        band_values = {}
        for Lmax, Lmin in bands_km:
            k1 = 1.0 / float(Lmax)
            k2 = 1.0 / float(Lmin)
            band_val = integrate_on_interval(kgrid, e_eke, k1, k2)
            band_values[(Lmax, Lmin)] = band_val
            rec[f"band_{Lmax}_{Lmin}km"] = band_val

        # Normalization by the sum of plotted bands
        eke_total = integrate_full_range(kgrid, e_eke)
        rec["eke_total"] = eke_total
        for Lmax, Lmin in bands_km:
            band_val = band_values[(Lmax, Lmin)]
            rec[f"frac_{Lmax}_{Lmin}km"] = (
                band_val / eke_total
                if np.isfinite(band_val) and np.isfinite(eke_total) and eke_total > 0
                else np.nan
            )
            
        cycle_records.append(rec)

    if not cycle_records:
        raise SystemExit("No cycle produced usable interpolated spectra.")

    df_cycles = pd.DataFrame(cycle_records)
    df_cycles = df_cycles.sort_values(["month_period", "cycle"]).reset_index(drop=True)

    frac_cols = [f"frac_{a}_{b}km" for a, b in bands_km]

    print("\nNaN count in fractional columns:")
    print(df_cycles[frac_cols].isna().sum())

    # --------------------------------------------------------
    # Average cycles falling in the same month
    # --------------------------------------------------------
    monthly = (
        df_cycles.groupby("month_period")[frac_cols]
        .mean()
        .reset_index()
        .sort_values("month_period")
    )

    monthly["frac_sum"] = monthly[frac_cols].sum(axis=1)
    print("\nMonthly sum of fractions (should be close to 1):")
    print(monthly[["month_period", "frac_sum"]])

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
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)

    for a, b in bands_km:
        col = f"frac_{a}_{b}km"
        ax.plot(
            monthly["month_period"],
            monthly[col],
            marker="o",
            lw=2.2,
            color=color_map[(a, b)],
            label=f"{a}–{b} km"
        )

    ax.set_ylabel(
        r"$EKE(k_1,k_2)/EKE_{tot}$")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, loc="upper right", frameon=True)

    # X ticks
    xticks = monthly["month_period"].tolist()
    xticklabels = [format_month_year_fr(pd.Timestamp(d)) for d in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")
    ax.set_xlabel("")
    ax.set_ylim(0,0.55)
    # No title
    ax.set_title("")

    # Save / show
    if args.outfile is not None:
        outpath = Path(args.outfile)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        print(f"\nFigure saved to: {outpath}")

    plt.show()


if __name__ == "__main__":
    main()