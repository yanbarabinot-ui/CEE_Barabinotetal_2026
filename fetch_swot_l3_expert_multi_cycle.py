#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 13:05:54 2025

@author: yan
"""

import subprocess
from pathlib import Path
import numpy as np
# list of cycles
cycles = np.arange(40,50,1)

# where to download files
base_outdir = Path("swot_l3_multi_cycle")
base_outdir.mkdir(exist_ok=True)

for c in cycles:
    cycle_str = f"{c:03d}"  # formate 1 = "001"
    outdir = base_outdir / f"cycle_{cycle_str}"
    figfile = base_outdir / f"med_c{cycle_str}.png"

    cmd = [
        "python", "fetch_swot_l3_expert_cycle_med.py",
        "--cycle", str(c),
        "--outdir", str(outdir),
        "--savefig", str(figfile),
        "--passes-file", "passes_region_fast_sampling.txt",
        "--show"
    ]

    print(f"\n=== Lancement du cycle {c} ===")
    subprocess.run(cmd, check=True)
