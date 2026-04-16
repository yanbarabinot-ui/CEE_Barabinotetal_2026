#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 15:35:11 2025

@author: yan
"""

import sys
import re
import csv
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ========= input =========
cycles = list(range(1, 49))                     
passes_file = "passes_region.txt"              
base_outdir = Path("swot_l3_multi_cycles_v3")     # output
base_outdir.mkdir(exist_ok=True)

# fetch (map per cycle)
save_fig_fetch = False                         
show_fetch_fig = False                         

# EKE binned
res = 0.025                                    # bins (~2 km)
min_count = 1                                  
cmap = "jet"
vmin, vmax = None, None                        
show_ke_fig = False

# bathymetry 
topo_file = "etopo2.nc"    # Smith and Sandwell 1997
topo_cut  = -0          # depth to filter data above a certain depth

# ========= some functions =========
PY = sys.executable  

RE_STATS = re.compile(
    r"\[STATS\]\s*mean=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+"
    r"std=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+"
    r"se=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)

def run_cmd(cmd, cwd=None):
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    return proc.returncode, proc.stdout, proc.stderr

# ========= Boucle principale =========
rows = []  # (cycle, mean, std, se)

for c in cycles:
    cyc = f"{c:03d}"
    outdir = base_outdir / f"cycle_{cyc}"
    outdir.mkdir(parents=True, exist_ok=True)

    """
     #1) FETCH : télécharge les .nc (skip si déjà présents)
    fetch_cmd = [
        PY, "fetch_swot_l3_expert_cycle.py",
        "--cycle", str(c),
        "--outdir", str(outdir),
    ]
    if passes_file:
        fetch_cmd += ["--passes-file", passes_file]
    if save_fig_fetch:
        fetch_cmd += ["--savefig", str(base_outdir / f"ssha_c{cyc}.png")]
    if show_fetch_fig:
        fetch_cmd += ["--show"]

    print(f"\n=== Cycle {c} : fetch ===")
    rc, out, err = run_cmd(fetch_cmd)
    if rc != 0:
        print(out)
        print(err)
        print(f"[WARN] fetch a échoué pour le cycle {c} — on continue avec les cycles suivants.")
        continue
    """
    # 2) EKE BINNED : calcule mean/std régionaux à partir des .nc du dossier
    ke_fig = base_outdir / f"v3_regional_ke_topo_0_c{cyc}.png"
    ke_cmd = [
        PY, "compute_med_ke_binned.py",
        "--indir", str(outdir),
        "--res", str(res),
        "--min-count", str(min_count),
        "--cmap", cmap,
    ]
    if vmin is not None:
        ke_cmd += ["--vmin", str(vmin)]
    if vmax is not None:
        ke_cmd += ["--vmax", str(vmax)]
    if save_fig_ke:
        ke_cmd += ["--savefig", str(ke_fig)]
    if show_ke_fig:
        ke_cmd += ["--show"]

    if (topo_file is not None) and (topo_cut is not None):
        ke_cmd += ["--topo-file", str(topo_file), "--topo", str(topo_cut)]

    print(f"=== Cycle {c} : compute KE ===")
    rc, out, err = run_cmd(ke_cmd)

    print(out)
    if rc != 0:
        print(err)
        print(f"[WARN] compute_med_ke_binned a échoué pour le cycle {c} — on passe au suivant.")
        continue

    # 3) PARSE : extract mean, std, se
    line_stats = next((ln for ln in out.splitlines() if ln.startswith("[STATS]")), None)
    if line_stats is None:
        print(f"[WARN] Impossible de trouver la ligne [STATS] pour le cycle {c}. Sortie stdout ci-dessous:")
        print(out)
        continue

    m = RE_STATS.search(line_stats)
    if not m:
        print(f"[WARN] Format inattendu de la ligne [STATS] pour le cycle {c}: {line_stats}")
        continue

    mean = float(m.group(1))
    std  = float(m.group(2))
    se   = float(m.group(3))

    # Garde-fou simple (KE en Med ~ 1e-5 à 1e-1 ; std rarement > 0.5)
    def sane(x): return np.isfinite(x) and (0 <= x < 1.0)
    if not (sane(mean) and sane(std)):
        print(f"[WARN] Valeurs suspectes cycle {c}: mean={mean}, std={std}. Ligne: {line_stats}")
        continue

    rows.append({"cycle": c, "mean_ke": mean, "std_ke": std, "se_mean": se})


# ========= output CSV and plots =========
csv_path = base_outdir / "v3_regional_ke_stats_topo_0.csv"
with csv_path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["cycle", "mean_ke", "std_ke", "se_mean"])
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"\nStats sauvegardées : {csv_path}")

if rows:
    rows = sorted(rows, key=lambda r: r["cycle"])
    xs   = [r["cycle"] for r in rows]
    mu   = [r["mean_ke"] for r in rows]
    sd   = [r["std_ke"]  for r in rows]
    se   = [r["se_mean"] if r["se_mean"] is not None else 0.0 for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.errorbar(xs, mu, yerr=sd, fmt="-o", capsize=3, label="Regional mean KE ± std (area-weighted)")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("EKE (m² s⁻²)")
    ax.set_title("SWOT L3 Expert — Regional EKE over Med")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax2 = ax.twinx()
    ax2.plot(xs, sd, "--s", label="Regional std EKE", alpha=0.7)
    ax2.set_ylabel("Std EKE (m² s⁻²)")
    ax2.legend(loc="lower right")

    out_png = base_outdir / "med_ke_vs_cycle.png"
    fig.savefig(out_png, dpi=180)
    print(f"Figure sauvegardée : {out_png}")
    plt.show()
else:
    print("Aucune statistique collectée — vérifie les logs ci-dessus.")

