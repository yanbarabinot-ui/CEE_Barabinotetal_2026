#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 15:56:20 2025

@author: yan
"""

"""
Figure 2 panels:
(1) EKE time series (with uncertainty envelopes)
(2) Skewness time series
Shared x-axis (date_median)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter

# -----------------------------
# 1) CHEMINS (EKE)
# -----------------------------
l3_dir = Path("swot_l3_multi_cycles")
csv1_path = l3_dir / "regional_ke_stats_med.csv"   # SWOT L3 v2.0.1

csv2_path = Path("compare_ke_l3_l4.csv")           # SWOT L4 MIOST + bornes de date
csv3_path = Path("compare_ke_l3_duacs_l4.csv")     # DUACS L4
csv4_path = l3_dir / "regional_ke_stats_topo_50.csv"     # SWOT L3 topo 50
csv5_path = l3_dir / "regional_ke_stats_topo_100.csv"     # SWOT L3 topo 100
csv6_path = l3_dir / "regional_ke_stats_topo_300.csv"     # SWOT L3 topo 300
csv7_path = Path("compare_ke_l3_l4_swot_like.csv")     # SWOT L4 MIOST swot-like
csv8_path = Path("compare_ke_l3_duacs_l4_swot_like.csv")     # DUACS L4

l3_dir = Path("swot_l3_multi_cycles_v3")
csv1bis_path = l3_dir / "v3_regional_ke_stats_topo_0.csv"   # SWOT L3 v3

fig_out = Path("eke_skewness_2panels.png")

# -----------------------------
# 2) LECTURE DES CSV (EKE)
# -----------------------------
df1 = pd.read_csv(csv1_path)
df1 = df1.rename(columns={"mean_ke": "mean_ke_l3", "se_mean": "se_ke_l3"})
if "cycle" in df1.columns:
    df1["cycle"] = df1["cycle"].astype(int)

df1bis = pd.read_csv(csv1bis_path)
df1bis = df1bis.rename(columns={"mean_ke": "mean_ke_l3_v3", "se_mean": "se_ke_l3_v3"})
if "cycle" in df1bis.columns:
    df1bis["cycle"] = df1bis["cycle"].astype(int)

df4 = pd.read_csv(csv4_path)
df4 = df4.rename(columns={"mean_ke": "mean_ke_l3_topo_50", "se_mean": "se_ke_l3_topo_50"})
if "cycle" in df4.columns:
    df4["cycle"] = df4["cycle"].astype(int)

df5 = pd.read_csv(csv5_path)
df5 = df5.rename(columns={"mean_ke": "mean_ke_l3_topo_100", "se_mean": "se_ke_l3_topo_100"})
if "cycle" in df5.columns:
    df5["cycle"] = df5["cycle"].astype(int)

df6 = pd.read_csv(csv6_path)
df6 = df6.rename(columns={"mean_ke": "mean_ke_l3_topo_300", "se_mean": "se_ke_l3_topo_300"})
if "cycle" in df6.columns:
    df6["cycle"] = df6["cycle"].astype(int)

df2 = pd.read_csv(csv2_path, parse_dates=["date_min", "date_max"])
if "cycle" in df2.columns:
    df2["cycle"] = df2["cycle"].astype(int)
df2["date_median"] = df2["date_min"] + (df2["date_max"] - df2["date_min"]) / 2.0
df2 = df2.rename(columns={"mean_ke_l4": "mean_ke_l4_miost"})

df7 = pd.read_csv(csv7_path, parse_dates=["date_min", "date_max"])
if "cycle" in df7.columns:
    df7["cycle"] = df7["cycle"].astype(int)
df7["date_median"] = df7["date_min"] + (df7["date_max"] - df7["date_min"]) / 2.0
df7 = df7.rename(columns={"mean_ke_l4": "mean_ke_l4_miost_swotlike", "se_ke_l4": "se_ke_L4_swotlike"})

df3 = pd.read_csv(csv3_path, parse_dates=["date_min", "date_max"])
if "cycle" in df3.columns:
    df3["cycle"] = df3["cycle"].astype(int)
df3 = df3.rename(columns={"mean_ke": "mean_ke_duacs", "se_ke": "se_ke_duacs"})

df8 = pd.read_csv(csv8_path, parse_dates=["date_min", "date_max"])
if "cycle" in df8.columns:
    df8["cycle"] = df8["cycle"].astype(int)
df8["date_median"] = df8["date_min"] + (df8["date_max"] - df8["date_min"]) / 2.0
df8 = df8.rename(columns={"mean_ke": "mean_ke_duacs_swotlike", "se_ke": "se_ke_duacs_swotlike"})

# -----------------------------
# 3) ALIGNEMENT PAR CYCLE (EKE)
# -----------------------------
base = df8[["cycle", "date_median"]].copy()
base = base.merge(df1[["cycle", "mean_ke_l3", "se_ke_l3"]], on="cycle", how="left")
base = base.merge(df1bis[["cycle", "mean_ke_l3_v3", "se_ke_l3_v3"]], on="cycle", how="left")
base = base.merge(df2[["cycle", "mean_ke_l4_miost", "se_ke_l4"]], on="cycle", how="left")
base = base.merge(df3[["cycle", "mean_ke_duacs", "se_ke_duacs"]], on="cycle", how="left")
base = base.merge(df4[["cycle", "mean_ke_l3_topo_50", "se_ke_l3_topo_50"]], on="cycle", how="left")
base = base.merge(df5[["cycle", "mean_ke_l3_topo_100", "se_ke_l3_topo_100"]], on="cycle", how="left")
base = base.merge(df6[["cycle", "mean_ke_l3_topo_300", "se_ke_l3_topo_300"]], on="cycle", how="left")
base = base.merge(df7[["cycle", "mean_ke_l4_miost_swotlike", "se_ke_L4_swotlike"]], on="cycle", how="left")
base = base.merge(df8[["cycle", "mean_ke_duacs_swotlike", "se_ke_duacs_swotlike"]], on="cycle", how="left")

base = base.sort_values("date_median").reset_index(drop=True)

# Conversion m² s⁻² → cm² s⁻² = × 1e4
ke_cols = [c for c in base.columns if "mean_ke" in c]
se_cols = [c for c in base.columns if "se_ke" in c]
for c in ke_cols:
    base[c] = base[c] * 1e4
for c in se_cols:
    base[c] = base[c] * 1e4

# -----------------------------
# 4) SKEWNESS (fonctions + chargements)
# -----------------------------
def skew_from_pdf(x, p):
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)

    p = np.clip(p, 0, np.inf)

    Z = np.trapz(p, x)
    if Z <= 0 or not np.isfinite(Z):
        return np.nan
    p = p / Z

    mu = np.trapz(x * p, x)
    var = np.trapz((x - mu) ** 2 * p, x)
    if var <= 0 or not np.isfinite(var):
        return np.nan
    sigma = np.sqrt(var)

    m3 = np.trapz((x - mu) ** 3 * p, x)
    return m3 / sigma**3

def load_skewness(csv_path):
    df = pd.read_csv(csv_path)
    cycles = []
    skews = []
    for c, sub in df.groupby("cycle"):
        x = sub["bin_center"].values
        p = sub["pdf"].values
        gamma1 = skew_from_pdf(x, p)
        cycles.append(c)
        skews.append(gamma1)
    return np.array(cycles), np.array(skews)

def load_cycle_dates(cycle_info_csv):
    path = Path(cycle_info_csv)
    df = pd.read_csv(path, parse_dates=["date_min", "date_max"])
    if "cycle" in df.columns:
        df["cycle"] = df["cycle"].astype(int)
    df["date_median"] = df["date_min"] + (df["date_max"] - df["date_min"]) / 2.0
    return df[["cycle", "date_median"]]

cycles_L3,    skew_L3    = load_skewness("derived_metrics/pdf_vort_norm.csv")
cycles_L3_topo,    skew_L3_topo    = load_skewness("derived_metrics_100/pdf_vort_norm.csv")
cycles_L3_v3, skew_L3_v3 = load_skewness("derived_metrics_v3/pdf_vort_norm.csv")
cycles_L4,    skew_L4    = load_skewness("derived_metrics_l4/pdf_vort_norm.csv")
cycles_DUACS, skew_DUACS = load_skewness("derived_metrics_l4_duacs/pdf_vort_norm.csv")

cycle_info = load_cycle_dates("compare_ke_l3_duacs_l4_swot_like.csv")

df_L3    = pd.DataFrame({"cycle": cycles_L3,    "skew": skew_L3})
df_L3_topo    = pd.DataFrame({"cycle": cycles_L3_topo,    "skew": skew_L3_topo})

df_L3_v3 = pd.DataFrame({"cycle": cycles_L3_v3, "skew": skew_L3_v3})
df_L4    = pd.DataFrame({"cycle": cycles_L4,    "skew": skew_L4})
df_DUACS = pd.DataFrame({"cycle": cycles_DUACS, "skew": skew_DUACS})

for df in (df_L3, df_L3_v3, df_L4, df_DUACS):
    df["cycle"] = df["cycle"].astype(int)

df_L3    = df_L3.merge(cycle_info, on="cycle", how="left").dropna(subset=["date_median"]).sort_values("date_median")
df_L3_topo    = df_L3_topo.merge(cycle_info, on="cycle", how="left").dropna(subset=["date_median"]).sort_values("date_median")
df_L3_v3 = df_L3_v3.merge(cycle_info, on="cycle", how="left").dropna(subset=["date_median"]).sort_values("date_median")
df_L4    = df_L4.merge(cycle_info, on="cycle", how="left").dropna(subset=["date_median"]).sort_values("date_median")
df_DUACS = df_DUACS.merge(cycle_info, on="cycle", how="left").dropna(subset=["date_median"]).sort_values("date_median")

# -----------------------------
# 5) FIGURE 2 PANELS (x partagé)
# -----------------------------
fig, (ax1, ax2) = plt.subplots(
    nrows=2, ncols=1,
    figsize=(10, 8), dpi=300,
    sharex=True,
    gridspec_kw={"height_ratios": [1.0, 1.0]}  # ax1 un peu plus grand
)

# Palette color-blind safe (Okabe & Ito) — identique code 1
colors = [
    "#0072B2",  # blue
    "#D55E00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # green
    "#F0E442",  # yellow
    "#CC79A7",  # vermillion/magenta
]

# ---- Panel 1: EKE (mêmes courbes que code 1)
def plot_err(ax, ycol, yerrcol, label, marker, color, alpha_fill=0.18):
    m = base[ycol].notna()
    x = base.loc[m, "date_median"].values
    y = base.loc[m, ycol].values

    if (yerrcol in base.columns) and base.loc[m, yerrcol].notna().any():
        e = base.loc[m, yerrcol].values
        ax.fill_between(
            x, y - e, y + e,
            color=color,
            alpha=alpha_fill,
            linewidth=0,
            zorder=1,
        )

    ax.plot(
        x, y,
        marker,
        color=color,
        linewidth=2,
        markersize=6,
        label=label,
        zorder=2,
    )

#plot_err(ax1, "mean_ke_l3",          "se_ke_l3",
#         "SWOT L3 v2.0.1",             "o-", colors[0])

#plot_err(ax1, "mean_ke_l3_topo_100", "se_ke_l3_topo_100",
#         "SWOT L3 v2.0.1 (<-100 m)", "+-", colors[0])

plot_err(ax1, "mean_ke_duacs_swotlike", "se_ke_duacs_swotlike",
         "DUACS L4",    "o-", colors[5])

plot_err(ax1, "mean_ke_l3_v3",          "se_ke_l3_v3",
         "SWOT L3 v3",             "o-", colors[3])

ax1.set_ylabel("EKE [cm$^2$ s$^{-2}$]", fontsize=12)
ax1.set_ylim(0, 300)
#ax1.legend(fontsize=12, loc='lower right', ncols=2)
# (grid désactivé comme dans ton code 1)
ax1.grid()
# ---- Panel 2: Skewness (mêmes paramètres que code 2)
color_L3    = "#0072B2"  # SWOT L3 v2.0.1
color_L3_v3 = "#009E73"  # SWOT L3 v3
color_L4    = "#F0E442"  # MIOST L4
color_DUACS = "#CC79A7"  # DUACS L4

#ax2.plot(
#    df_L3["date_median"], df_L3["skew"],
#    "-o",
#    label="SWOT L3 v2.0.1",
#    linewidth=2,
#    color=color_L3,
#)

#ax2.plot(
#    df_L3_topo["date_median"], df_L3_topo["skew"],
#    "-+",
#    label="SWOT L3 v2.0.1 (<-100 m)",
#    linewidth=2,
#    color=color_L3,
#)


ax2.plot(
    df_L3_v3["date_median"], df_L3_v3["skew"],
    "-o",
    label="SWOT L3 v3",
    linewidth=2,
    color=color_L3_v3,
)

#ax2.plot(
#    df_L4["date_median"], df_L4["skew"],
#    "-s",
#    label="MIOST L4",
#    linewidth=2,
#    color=color_L4,
#)

ax2.plot(
    df_DUACS["date_median"], df_DUACS["skew"],
    "-^",
    label="DUACS L4",
    linewidth=2,
    color=color_DUACS,
)
ax2.grid()
ax2.axhline(0, color="k", linestyle="--", linewidth=0.8)
ax2.set_ylabel("Skewness (PDF $\zeta/f$)", fontsize=12)
ax2.set_xlabel("")  # comme tes codes
ax2.legend(ncols=2, fontsize=12, loc='lower right')
ax2.set_ylim(-1, 1)

# ---- Axe des dates commun (appliqué sur ax2, et partagé avec ax1)
ax2.xaxis.set_major_locator(MonthLocator(interval=1))
ax2.xaxis.set_major_formatter(DateFormatter("%b %Y"))
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", fontsize=12)
plt.setp(ax1.get_yticklabels(), fontsize=12)
plt.setp(ax2.get_yticklabels(), fontsize=12)

fig.tight_layout()
fig.savefig(fig_out, bbox_inches="tight")
plt.show()

print("Figure sauvegardée sous :", fig_out.resolve())
