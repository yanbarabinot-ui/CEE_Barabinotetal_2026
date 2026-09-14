
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure EKE(k1,k2) by wavelength band with seasonal diagnostics.

Left column:
    Monthly EKE(k1,k2) for each wavelength band.

Middle column:
    SWOT-DUACS seasonal EKE difference for JJA and DJF.

Right column:
    Seasonal amplification 100 * (DJF - JJA) / JJA for SWOT and DUACS.

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

    if month not in monthly_plot["month_period"].values:
        new_row = {"month_period": month}
        for col in cols:
            new_row[col] = np.nan
        monthly_plot = pd.concat(
            [monthly_plot, pd.DataFrame([new_row])],
            ignore_index=True
        )

    monthly_plot = monthly_plot.sort_values("month_period").reset_index(drop=True)

    for col in cols:
        monthly_plot.loc[monthly_plot["month_period"] == month, col] = np.nan
        monthly_plot[col] = monthly_plot[col].interpolate(
            method="linear",
            limit_direction="both"
        )

    return monthly_plot


def add_djf_bands(ax, start_date, end_date):
    """Add discrete DJF shading."""
    if pd.isna(start_date) or pd.isna(end_date):
        return

    for year in range(start_date.year - 1, end_date.year + 1):
        djf_start = pd.Timestamp(year=year, month=12, day=1)
        djf_end = pd.Timestamp(year=year + 1, month=3, day=1)
        left = max(djf_start, start_date)
        right = min(djf_end, end_date)
        if left < right:
            ax.axvspan(
                left,
                right,
                facecolor="0.80",
                alpha=0.18,
                linewidth=0,
                zorder=0,
            )


def add_period_separation(
    ax,
    plot_start,
    plot_end,
    swot_start,
    swot_end,
    common_start,
    common_end,
):
    add_djf_bands(ax, plot_start, plot_end)

    if pd.notna(common_start) and pd.notna(swot_start) and swot_start < common_start:
        ax.axvspan(
            swot_start,
            common_start,
            facecolor="#EAF2F8",
            alpha=0.75,
            linewidth=0,
            zorder=0.1,
        )
        ax.axvline(
            common_start,
            color="0.45",
            linestyle="--",
            linewidth=1.0,
            zorder=0.5,
        )

    if pd.notna(common_end) and pd.notna(swot_end) and common_end < swot_end:
        ax.axvspan(
            common_end,
            swot_end,
            facecolor="#EAF2F8",
            alpha=0.75,
            linewidth=0,
            zorder=0.1,
        )
        ax.axvline(
            common_end,
            color="0.45",
            linestyle="--",
            linewidth=1.0,
            zorder=0.5,
        )


def compute_seasonal_increase_percent(monthly, bands_km, start=None, end=None):
    """
    Compute 100 * (EKE_DJF - EKE_JJA) / EKE_JJA.

    Calendar-month climatologies are used so that each month has the same
    weight. Plot-only interpolated months are deliberately excluded.
    """
    if monthly is None or monthly.empty:
        return {band: np.nan for band in bands_km}

    data = monthly.copy()
    if start is not None and pd.notna(start):
        data = data[data["month_period"] >= pd.Timestamp(start)]
    if end is not None and pd.notna(end):
        data = data[data["month_period"] <= pd.Timestamp(end)]

    results = {}

    for Lmax, Lmin in bands_km:
        col = f"eke_{Lmax}_{Lmin}km"
        valid = data[["month_period", col]].dropna()

        if valid.empty:
            results[(Lmax, Lmin)] = np.nan
            continue

        climatology = valid.groupby(valid["month_period"].dt.month)[col].mean()

        eke_djf = climatology.reindex([12, 1, 2]).mean()
        eke_jja = climatology.reindex([6, 7, 8]).mean()

        if (
            not np.isfinite(eke_djf)
            or not np.isfinite(eke_jja)
            or eke_jja == 0
        ):
            results[(Lmax, Lmin)] = np.nan
        else:
            results[(Lmax, Lmin)] = 100.0 * (eke_djf - eke_jja) / eke_jja

    return results


def mean_and_sem(values):
    """
    Return mean, standard error of the mean, and sample size.
    If only one valid value is available, the uncertainty is set to 0.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan, np.nan, 0

    mean = np.nanmean(values)
    if values.size < 2:
        sem = 0.0
    else:
        sem = np.nanstd(values, ddof=1) / np.sqrt(values.size)

    return mean, sem, int(values.size)


def filter_monthly_for_metrics(monthly, start=None, end=None):
    """Filter monthly diagnostics over the period used for seasonal metrics."""
    if monthly is None or monthly.empty:
        return None

    data = monthly.copy()
    if start is not None and pd.notna(start):
        data = data[data["month_period"] >= pd.Timestamp(start)]
    if end is not None and pd.notna(end):
        data = data[data["month_period"] <= pd.Timestamp(end)]

    return data


def compute_seasonal_increase_percent_and_error(monthly, bands_km, start=None, end=None):
    """
    Compute seasonal amplification and its uncertainty.

    The central value follows compute_seasonal_increase_percent: it is based on
    calendar-month climatologies, so June, July and August have the same weight
    for JJA, and December, January and February have the same weight for DJF.

    The error bar is obtained by propagating the standard errors of the DJF and
    JJA monthly samples:

        R = 100 * (DJF - JJA) / JJA

    assuming independent DJF and JJA means.
    """
    results = {
        band: {"value": np.nan, "error": np.nan, "jja_n": 0, "djf_n": 0}
        for band in bands_km
    }

    data = filter_monthly_for_metrics(monthly, start=start, end=end)
    if data is None or data.empty:
        return results

    for Lmax, Lmin in bands_km:
        col = f"eke_{Lmax}_{Lmin}km"
        valid = data[["month_period", col]].dropna()

        if valid.empty:
            continue

        month = valid["month_period"].dt.month
        climatology = valid.groupby(month)[col].mean()

        djf_mean = climatology.reindex([12, 1, 2]).mean()
        jja_mean = climatology.reindex([6, 7, 8]).mean()

        jja_values = valid.loc[month.isin([6, 7, 8]), col].values
        djf_values = valid.loc[month.isin([12, 1, 2]), col].values

        _, jja_sem, jja_n = mean_and_sem(jja_values)
        _, djf_sem, djf_n = mean_and_sem(djf_values)

        if (
            not np.isfinite(djf_mean)
            or not np.isfinite(jja_mean)
            or jja_mean == 0
        ):
            continue

        value = 100.0 * (djf_mean - jja_mean) / jja_mean

        if np.isfinite(jja_sem) and np.isfinite(djf_sem):
            error = np.sqrt(
                (100.0 * djf_sem / jja_mean) ** 2
                + (100.0 * djf_mean * jja_sem / (jja_mean ** 2)) ** 2
            )
        else:
            error = np.nan

        results[(Lmax, Lmin)] = {
            "value": value,
            "error": error,
            "jja_n": jja_n,
            "djf_n": djf_n,
        }

    return results


def compute_swot_duacs_seasonal_difference(monthly_swot, monthly_duacs, bands_km, start=None, end=None):
    """
    Compute seasonal mean SWOT-DUACS EKE difference for JJA and DJF.

    The central value is the mean monthly climatological difference for the
    season. The error bar is the standard error of the monthly paired
    SWOT-DUACS differences available during the selected period.
    """
    seasons = {
        "JJA": [6, 7, 8],
        "DJF": [12, 1, 2],
    }

    results = {
        band: {
            season: {"value": np.nan, "error": np.nan, "n": 0, "points": np.array([], dtype=float)}
            for season in seasons
        }
        for band in bands_km
    }

    if monthly_swot is None or monthly_duacs is None:
        return results

    swot = filter_monthly_for_metrics(monthly_swot, start=start, end=end)
    duacs = filter_monthly_for_metrics(monthly_duacs, start=start, end=end)

    if swot is None or duacs is None or swot.empty or duacs.empty:
        return results

    for Lmax, Lmin in bands_km:
        col = f"eke_{Lmax}_{Lmin}km"

        merged = pd.merge(
            swot[["month_period", col]].rename(columns={col: "swot"}),
            duacs[["month_period", col]].rename(columns={col: "duacs"}),
            on="month_period",
            how="inner",
        ).dropna(subset=["swot", "duacs"])

        if merged.empty:
            continue

        merged["diff"] = merged["swot"] - merged["duacs"]
        month = merged["month_period"].dt.month

        for season_name, season_months in seasons.items():
            season = merged.loc[month.isin(season_months)].copy()
            if season.empty:
                continue

            season_month = season["month_period"].dt.month
            climatology = season.groupby(season_month)["diff"].mean()
            value = climatology.reindex(season_months).mean()
            _, sem, n = mean_and_sem(season["diff"].values)

            results[(Lmax, Lmin)][season_name] = {
                "value": value,
                "error": sem,
                "n": n,
                # Individual paired monthly SWOT-DUACS differences used to
                # characterize the seasonal distribution and compute the SEM.
                "points": season["diff"].to_numpy(dtype=float),
            }

    return results


def annotate_bar_end(ax, value, y, xlim, err=0.0, suffix="%"):
    """
    Add label at the end of a horizontal bar, after the error bar.
    """
    if not np.isfinite(value):
        ax.text(
            0.0,
            y,
            "n/a",
            va="center",
            ha="left",
            fontsize=9,
            color="0.25",
        )
        return

    xspan = xlim[1] - xlim[0]
    offset = 0.025 * xspan
    err = err if np.isfinite(err) else 0.0

    if value >= 0:
        x_text = value + err + offset
        ha = "left"
    else:
        x_text = value - err - offset
        ha = "right"

    ax.text(
        x_text,
        y,
        f"{value:+.0f}{suffix}",
        va="center",
        ha=ha,
        fontsize=9,
        color="0.20",
    )


def annotate_vertical_bar(ax, value, x, ylim, err=0.0):
    """
    Add numeric label above or below a vertical bar, after the error bar.
    """
    if not np.isfinite(value):
        ax.text(
            x,
            0.0,
            "n/a",
            va="bottom",
            ha="center",
            fontsize=9,
            color="0.25",
            zorder=30,
            bbox=dict(
                boxstyle="round,pad=0.12",
                facecolor="white",
                edgecolor="black",
                linewidth=0.5,
                alpha=0.9,
            ),
        )
        return

    yspan = ylim[1] - ylim[0]
    offset = 0.035 * yspan
    err = err if np.isfinite(err) else 0.0

    if value >= 0:
        y_text = value + err + offset
        va = "bottom"
    else:
        y_text = value - err - offset
        va = "top"

    ax.text(
        x,
        y_text,
        f"{value:.0f}",
        va=va,
        ha="center",
        fontsize=9,
        color="0.20",
        clip_on=False,
        zorder=30,
        bbox=dict(
            boxstyle="round,pad=0.12",
            facecolor="white",
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
        ),
    )


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description=(
            "SWOT L3 v3 and DUACS monthly averaged band-limited EKE "
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
        help="Output figure path, for example figure.png or figure.pdf. If omitted, only display."
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

    bands_km = [
        (500, 200),
        (200, 100),
        (100, 60),
        (60, 20),
    ]

    eke_cols = [f"eke_{a}_{b}km" for a, b in bands_km]

    # --------------------------------------------------------
    # SWOT L3 v3
    # --------------------------------------------------------
    monthly = compute_monthly_band_fractions(df_spec, df_time, bands_km, gf2)
    if monthly is None:
        raise SystemExit("No SWOT L3 v3 monthly EKE could be computed.")

    print("\nMonths present in SWOT monthly:")
    print(monthly["month_period"].dt.strftime("%Y-%m").tolist())

    print("\nCycle/date mapping:")
    print(df_time[["cycle", "date_min", "date_max", "date_median", "month_period"]].to_string(index=False))

    # --------------------------------------------------------
    # DUACS
    # --------------------------------------------------------
    duacs_spec_csv = Path(args.duacs_dir) / "ssh_spectra.csv"
    df_duacs_spec = load_l3v3_spectra(duacs_spec_csv)

    monthly_duacs = compute_monthly_band_fractions(df_duacs_spec, df_time, bands_km, gf2)
    if monthly_duacs is None:
        print("[WARN] No DUACS monthly EKE could be computed.", file=sys.stderr)

    # --------------------------------------------------------
    # Temporal coverage and common SWOT-DUACS period
    # --------------------------------------------------------
    swot_valid = monthly.dropna(subset=eke_cols, how="all")
    duacs_valid = (
        monthly_duacs.dropna(subset=eke_cols, how="all")
        if monthly_duacs is not None
        else None
    )

    swot_start = swot_valid["month_period"].min()
    swot_end = swot_valid["month_period"].max()

    if duacs_valid is not None and not duacs_valid.empty:
        duacs_start = duacs_valid["month_period"].min()
        duacs_end = duacs_valid["month_period"].max()
        common_start = max(swot_start, duacs_start)
        common_end = min(swot_end, duacs_end)
    else:
        duacs_start = pd.NaT
        duacs_end = pd.NaT
        common_start = pd.NaT
        common_end = pd.NaT

    all_date_series = [swot_valid["month_period"]]
    if duacs_valid is not None and not duacs_valid.empty:
        all_date_series.append(duacs_valid["month_period"])

    all_dates = pd.concat(all_date_series, ignore_index=True).dropna()
    plot_start = all_dates.min()
    plot_end = all_dates.max()

    # Use the common period for a directly comparable seasonal metric.
    if pd.notna(common_start) and pd.notna(common_end) and common_start <= common_end:
        seasonal_start = common_start
        seasonal_end = common_end
    else:
        seasonal_start = swot_start
        seasonal_end = swot_end

    seasonal_swot_stats = compute_seasonal_increase_percent_and_error(
        monthly,
        bands_km,
        start=seasonal_start,
        end=seasonal_end,
    )
    seasonal_swot = {
        band: seasonal_swot_stats[band]["value"]
        for band in bands_km
    }

    # --------------------------------------------------------
    # DUACS seasonal metrics
    # Remove July 2024 only for the 60–100 km band. The same
    # DUACS metric table is used for the seasonal amplification
    # and the SWOT-DUACS seasonal EKE difference.
    # --------------------------------------------------------
    if monthly_duacs is not None:
        monthly_duacs_for_metrics = monthly_duacs.copy()

        col_60_100 = "eke_100_60km"  # band displayed as 60–100 km
        july_2024 = pd.Timestamp("2024-07-01")

        monthly_duacs_for_metrics.loc[
            monthly_duacs_for_metrics["month_period"] == july_2024,
            col_60_100
        ] = np.nan

        seasonal_duacs_stats = compute_seasonal_increase_percent_and_error(
            monthly_duacs_for_metrics,
            bands_km,
            start=seasonal_start,
            end=seasonal_end,
        )
        seasonal_duacs = {
            band: seasonal_duacs_stats[band]["value"]
            for band in bands_km
        }

        seasonal_diff_stats = compute_swot_duacs_seasonal_difference(
            monthly,
            monthly_duacs_for_metrics,
            bands_km,
            start=seasonal_start,
            end=seasonal_end,
        )
    else:
        monthly_duacs_for_metrics = None
        seasonal_duacs_stats = None
        seasonal_duacs = None
        seasonal_diff_stats = compute_swot_duacs_seasonal_difference(
            monthly,
            None,
            bands_km,
            start=seasonal_start,
            end=seasonal_end,
        )

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
    # Colors
    # --------------------------------------------------------
    swot_color = "#009E73"
    duacs_color = "#CC79A7"
    BAR_BLUE = "#0072B2"    # JJA, as in the reference ΔEKE bar plot
    BAR_ORANGE = "#E69F00"  # DJF, as in the reference ΔEKE bar plot

    # --------------------------------------------------------
    # Common y-limit for all left panels
    # --------------------------------------------------------
    ymax_vals = []

    for Lmax, Lmin in bands_km:
        col = f"eke_{Lmax}_{Lmin}km"
        ymax_vals.append(np.nanmax(monthly_plot[col].values))

        if monthly_duacs_plot is not None:
            ymax_vals.append(np.nanmax(monthly_duacs_plot[col].values))

    ymax = np.nanmax(ymax_vals)
    ylim = (0.0, 1.1 * ymax)

    # --------------------------------------------------------
    # Common x-limit for seasonal amplification bar plots
    # --------------------------------------------------------
    seasonal_extents = []

    for band in bands_km:
        swot_stat = seasonal_swot_stats.get(band, {})
        val_swot = swot_stat.get("value", np.nan)
        err_swot = swot_stat.get("error", np.nan)
        err_swot = err_swot if np.isfinite(err_swot) else 0.0
        if np.isfinite(val_swot):
            seasonal_extents.extend([val_swot - err_swot, val_swot + err_swot])

        if seasonal_duacs_stats is not None:
            duacs_stat = seasonal_duacs_stats.get(band, {})
            val_duacs = duacs_stat.get("value", np.nan)
            err_duacs = duacs_stat.get("error", np.nan)
            err_duacs = err_duacs if np.isfinite(err_duacs) else 0.0
            if np.isfinite(val_duacs):
                seasonal_extents.extend([val_duacs - err_duacs, val_duacs + err_duacs])

    seasonal_extents = np.asarray(seasonal_extents, dtype=float)
    finite_extents = seasonal_extents[np.isfinite(seasonal_extents)]

    if finite_extents.size > 0:
        xmin = min(0.0, np.nanmin(finite_extents))
        xmax = max(0.0, np.nanmax(finite_extents))
        xspan = xmax - xmin
        if xspan == 0:
            xspan = max(abs(xmax), 1.0)
        bar_xlim = (xmin - 0.25 * xspan, xmax + 0.35 * xspan)
    else:
        bar_xlim = (-10.0, 10.0)

    # --------------------------------------------------------
    # Common y-limit for SWOT-DUACS seasonal EKE difference
    # --------------------------------------------------------
    diff_extents = []

    for band in bands_km:
        for season_name in ["JJA", "DJF"]:
            stat = seasonal_diff_stats.get(band, {}).get(season_name, {})
            value = stat.get("value", np.nan)
            error = stat.get("error", np.nan)
            error = error if np.isfinite(error) else 0.0
            if np.isfinite(value):
                diff_extents.extend([value - error, value + error])

    diff_extents = np.asarray(diff_extents, dtype=float)
    finite_diff_extents = diff_extents[np.isfinite(diff_extents)]

    if finite_diff_extents.size > 0:
        ymin = min(0.0, np.nanmin(finite_diff_extents))
        ymax_diff = max(0.0, np.nanmax(finite_diff_extents))
        yspan = ymax_diff - ymin
        if yspan == 0:
            yspan = max(abs(ymax_diff), 1.0)
        lower_pad = 0.10 * yspan if ymin < 0 else 0.0
        upper_pad = 0.2 * yspan
        diff_ylim = (ymin - lower_pad, ymax_diff + upper_pad)
    else:
        diff_ylim = (-1.0, 1.0)

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------
    fig = plt.figure(
        figsize=(11, 10),
        constrained_layout=True,
    )
    gs = fig.add_gridspec(
        4,
        3,
        width_ratios=[5, 1, 1],
        height_ratios=[1, 1, 1, 1],
        wspace=0.10,
    )

    axes_ts = np.empty(4, dtype=object)
    axes_diff = np.empty(4, dtype=object)
    axes_bar = np.empty(4, dtype=object)

    for i in range(4):
        if i == 0:
            axes_ts[i] = fig.add_subplot(gs[i, 0])
            axes_diff[i] = fig.add_subplot(gs[i, 1])
            axes_bar[i] = fig.add_subplot(gs[i, 2])
        else:
            axes_ts[i] = fig.add_subplot(gs[i, 0], sharex=axes_ts[0])
            axes_diff[i] = fig.add_subplot(
                gs[i, 1],
                sharex=axes_diff[0],
                sharey=axes_diff[0],
            )
            axes_bar[i] = fig.add_subplot(gs[i, 2], sharex=axes_bar[0])

    # Add background separation to left column
    for ax in axes_ts:
        add_period_separation(
            ax,
            plot_start,
            plot_end,
            swot_start,
            swot_end,
            common_start,
            common_end,
        )
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(*ylim)
        ax.set_xlim(
            plot_start - pd.Timedelta(days=15),
            plot_end + pd.Timedelta(days=15),
        )

    # SWOT-only text, shown only on the first left panel
    if pd.notna(common_start) and pd.notna(swot_start) and swot_start < common_start:
        left_mid = swot_start + (common_start - swot_start) / 2
        axes_ts[0].text(
            left_mid,
            0.98,
            "SWOT only",
            transform=axes_ts[0].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
            color="0.30",
        )

    if pd.notna(common_end) and pd.notna(swot_end) and common_end < swot_end:
        right_mid = common_end + (swot_end - common_end) / 2
        axes_ts[0].text(
            right_mid,
            0.98,
            "SWOT only",
            transform=axes_ts[0].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
            color="0.30",
        )

    # --------------------------------------------------------
    # Plot each wavelength band on a separate row
    # --------------------------------------------------------
    error_kw = {
        "ecolor": "black",
        "elinewidth": 1.0,
        "capsize": 3,
        "capthick": 1.0,
    }

    for i, (Lmax, Lmin) in enumerate(bands_km):
        ax = axes_ts[i]
        axd = axes_diff[i]
        axb = axes_bar[i]

        col = f"eke_{Lmax}_{Lmin}km"
        band_label = f"{Lmin}–{Lmax} km"

        # SWOT
        ax.plot(
            monthly_plot["month_period"],
            monthly_plot[col],
            marker="o",
            lw=2.2,
            color=swot_color,
            label="SWOT",
        )

        # DUACS
        if monthly_duacs_plot is not None:
            ax.plot(
                monthly_duacs_plot["month_period"],
                monthly_duacs_plot[col],
                marker="o",
                lw=2.0,
                color=duacs_color,
                label="DUACS",
            )

        ax.set_title(band_label, fontweight="bold", loc="left")

        if i == 0:
            ax.legend(
                loc="upper left",
                frameon=True,
                ncol=2,
            )

        ax.set_ylabel(r"$EKE(k_1,k_2)$" "\n" r"[cm$^2$ s$^{-2}$]")

        # ----------------------------------------------------
        # Top right sub-panel: SWOT-DUACS seasonal EKE difference
        # ----------------------------------------------------
        diff_jja = seasonal_diff_stats[(Lmax, Lmin)]["JJA"]["value"]
        diff_djf = seasonal_diff_stats[(Lmax, Lmin)]["DJF"]["value"]
        diff_jja_err = seasonal_diff_stats[(Lmax, Lmin)]["JJA"]["error"]
        diff_djf_err = seasonal_diff_stats[(Lmax, Lmin)]["DJF"]["error"]
        diff_jja_points = seasonal_diff_stats[(Lmax, Lmin)]["JJA"]["points"]
        diff_djf_points = seasonal_diff_stats[(Lmax, Lmin)]["DJF"]["points"]

        x_pos = np.array([0, 1])
        diff_values = np.array([diff_jja, diff_djf], dtype=float)
        diff_errors = np.array([diff_jja_err, diff_djf_err], dtype=float)
        diff_errors_plot = np.where(np.isfinite(diff_errors), diff_errors, 0.0)

        axd.axhline(0, color="0.35", lw=1.0, zorder=0)
        axd.bar(
            x_pos,
            diff_values,
            yerr=diff_errors_plot,
            color=[BAR_BLUE, BAR_ORANGE],
            width=0.65,
            alpha=0.95,
            error_kw=error_kw,
            zorder=2,
        )

        # Show the individual paired monthly SWOT-DUACS differences underlying
        # each seasonal summary. A deterministic horizontal jitter prevents
        # overlapping points without changing their data values.
        for xx, points, point_color in [
            (0, diff_jja_points, BAR_BLUE),
            (1, diff_djf_points, BAR_ORANGE),
        ]:
            points = np.asarray(points, dtype=float)
            points = points[np.isfinite(points)]
            if points.size:
                if points.size == 1:
                    jitter = np.array([0.0])
                else:
                    jitter = np.linspace(-0.16, 0.16, points.size)

                axd.scatter(
                    xx + jitter,
                    points,
                    s=26,
                    facecolor=point_color,
                    edgecolor="black",
                    linewidth=0.6,
                    alpha=0.95,
                    zorder=5,
                    clip_on=False,
                )

        axd.set_xticks(x_pos)
        axd.set_xticklabels(["JJA", "DJF"])
        axd.set_ylim(*diff_ylim)
        axd.grid(True, axis="y", alpha=0.25)
        axd.set_axisbelow(True)
        axd.set_ylabel(r"$\Delta$EKE [cm$^2$ s$^{-2}$]", fontsize=10)
        axd.tick_params(axis="both", labelsize=10)

        if i == 0:
            axd.set_title(
                "SWOT-DUACS",
                fontweight="bold",
                fontsize=11,
            )

        for val, err, xx in zip(diff_values, diff_errors, x_pos):
            annotate_vertical_bar(axd, val, xx, diff_ylim, err=err)

        # ----------------------------------------------------
        # Bottom right sub-panel: 100 * (DJF - JJA) / JJA
        # ----------------------------------------------------
        swot_stat = seasonal_swot_stats[(Lmax, Lmin)]
        swot_pct = swot_stat["value"]
        swot_pct_err = swot_stat["error"]

        if seasonal_duacs_stats is not None:
            duacs_stat = seasonal_duacs_stats[(Lmax, Lmin)]
            duacs_pct = duacs_stat["value"]
            duacs_pct_err = duacs_stat["error"]
        else:
            duacs_pct = np.nan
            duacs_pct_err = np.nan

        x_pos = np.array([0, 1])
        values = np.array([swot_pct, duacs_pct], dtype=float)
        errors = np.array([swot_pct_err, duacs_pct_err], dtype=float)
        errors_plot = np.where(np.isfinite(errors), errors, 0.0)
        colors = [swot_color, duacs_color]
        
        axb.axhline(0, color="0.35", lw=1.0, zorder=0)
        axb.bar(
            x_pos,
            values,
            yerr=errors_plot,
            color=colors,
            width=0.65,
            alpha=0.95,
            error_kw=error_kw,
        )

        axb.set_xticks(x_pos)
        axb.set_xticklabels(["SWOT", "DUACS"])
        axb.set_ylim(-50,90)
        axb.grid(True, axis="y", alpha=0.25)
        axb.set_axisbelow(True)
        axb.tick_params(axis="both", labelsize=10)

        if i == 0:
            axb.set_title(
                "Seasonal amplification [%]",
                fontweight="bold",
                fontsize=11,
            )

        for val, err, xx in zip(values, errors, x_pos):
            annotate_vertical_bar(axb, val, xx, bar_xlim, err=err)
    
    # --------------------------------------------------------
    # X-axis formatting
    # --------------------------------------------------------
    for ax in axes_ts[:-1]:
        ax.tick_params(labelbottom=False)

    axes_ts[-1].xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
    axes_ts[-1].xaxis.set_major_formatter(DateFormatter("%b\n%Y"))
    plt.setp(axes_ts[-1].get_xticklabels(), rotation=0, ha="center")
    axes_ts[-1].set_xlabel("")

    for axd in axes_diff[:-1]:
        axd.tick_params(labelbottom=False)

    axes_diff[-1].set_xlabel("season")

    for axb in axes_bar[:-1]:
        axb.tick_params(labelbottom=False)

    axes_bar[-1].set_xlabel("product")

    # --------------------------------------------------------
    # Print seasonal values
    # --------------------------------------------------------
    print("\nSeasonal amplification = 100 * (EKE_DJF - EKE_JJA) / EKE_JJA")
    print(f"Period used: {seasonal_start:%Y-%m} to {seasonal_end:%Y-%m}")

    for Lmax, Lmin in bands_km:
        swot_stat = seasonal_swot_stats[(Lmax, Lmin)]
        if seasonal_duacs_stats is not None:
            duacs_stat = seasonal_duacs_stats[(Lmax, Lmin)]
            print(
                f"{Lmin:>3}–{Lmax:<3} km: "
                f"SWOT={swot_stat['value']:+.1f} ± {swot_stat['error']:.1f}%, "
                f"DUACS={duacs_stat['value']:+.1f} ± {duacs_stat['error']:.1f}%"
            )
        else:
            print(
                f"{Lmin:>3}–{Lmax:<3} km: "
                f"SWOT={swot_stat['value']:+.1f} ± {swot_stat['error']:.1f}%"
            )

    print("\nSWOT-DUACS seasonal EKE difference")
    for Lmax, Lmin in bands_km:
        jja = seasonal_diff_stats[(Lmax, Lmin)]["JJA"]
        djf = seasonal_diff_stats[(Lmax, Lmin)]["DJF"]
        print(
            f"{Lmin:>3}–{Lmax:<3} km: "
            f"JJA={jja['value']:+.1f} ± {jja['error']:.1f} cm^2 s^-2, "
            f"DJF={djf['value']:+.1f} ± {djf['error']:.1f} cm^2 s^-2"
        )

    # --------------------------------------------------------
    # Save / show
    # --------------------------------------------------------
    if args.outfile is not None:
        outpath = Path(args.outfile)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        print(f"\nFigure saved to: {outpath}")

    plt.show()


if __name__ == "__main__":
    main()