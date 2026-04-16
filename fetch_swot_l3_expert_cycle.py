from __future__ import annotations

"""
Created on Mon Oct 13 10:24:44 2025

@author: yan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download SWOT L3 expert files for a given cycle

how to use:
  python fetch_swot_l3_expert_cycle.py --cycle 2 --passes-file passes_region.txt \
      --lat-min 30 --lat-max 46 --lon-min -6 --lon-max 36 \
      --outdir data_swot_c002 --show

"""

import argparse
import os
import sys
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Set
from urllib.parse import urljoin
import netrc as netrc_mod

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from lxml import etree
from tqdm import tqdm
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import cm, colors

# --- Constantes catalogue ---
BASE_TDS = "https://tds-odatis.aviso.altimetry.fr/thredds/"
#DATASET_ROOT = "dataset-l3-swot-karin-nadir-validated/l3_lr_ssh/v2_0_1/Expert/" #for V2.0.1
DATASET_ROOT = "dataset-l3-swot-karin-nadir-validated/l3_lr_ssh/v3_0/Expert/forward/" #for V3.0
CATALOG_XML = urljoin(BASE_TDS, f"catalog/{DATASET_ROOT}cycle_{{cycle:03d}}/catalog.xml")
FILESERVER_BASE = urljoin(BASE_TDS, "fileServer/")


# --- Outils auth/session -----------------------------------------------------------------

def build_session(username: Optional[str], password: Optional[str]) -> requests.Session:
    sess = requests.Session()
    # Retries robustes
    retries = Retry(
        total=5, backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["HEAD", "GET", "OPTIONS"])
    )
    sess.mount("https://", HTTPAdapter(max_retries=retries))
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    if username is None or password is None:
        try:
            nrc = netrc_mod.netrc()
            auth = nrc.authenticators("tds-odatis.aviso.altimetry.fr")
            if auth:
                sess.auth = (auth[0], auth[2])
        except FileNotFoundError:
            pass
    if username is not None and password is not None:
        sess.auth = (username, password)

    return sess


# --- Parsing du catalog.xml ---------------------------------------------------------------

def list_cycle_datasets(cycle: int, session: requests.Session) -> List[Tuple[str, str]]:
    url = CATALOG_XML.format(cycle=cycle)
    r = session.get(url, timeout=30)
    r.raise_for_status()

    # Parse XML
    root = etree.fromstring(r.content)
    ns = {
        "cat": "http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0"
    }

    results = []
    # Les datasets sont des <dataset ... urlPath="...nc" name="...nc">
    for ds in root.xpath("//cat:dataset[@urlPath]", namespaces=ns):
        url_path = ds.attrib.get("urlPath", "")
        name = ds.attrib.get("name", "")
        if name.endswith(".nc") or url_path.endswith(".nc"):
            results.append((name, url_path))
    # tri par nom pour reproductibilité
    results.sort(key=lambda t: t[0])
    return results


def parse_cycle_pass_from_name(name: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract (cycle, pass) depuis un nom de fichier du type :
    'SWOT_L3_LR_SSH_Expert_002_001_20230811T021853_...nc' -> (2, 1)
    """
    m = re.search(r"_(\d{3})_(\d{3})_", name)
    if not m:
        return None, None
    cyc = int(m.group(1))
    pas = int(m.group(2))
    return cyc, pas


# --- Télécharger en streaming ------------------------------------------------------------

def download_fileserver_file(url_path: str, outdir: Path, session: requests.Session) -> Path:
    """
    dowload a file from /thredds/fileServer/<urlPath> 
    """
    url = urljoin(FILESERVER_BASE, url_path)
    filename = Path(url_path).name
    outpath = outdir / filename

    # Skip si déjà présent
    if outpath.exists() and outpath.stat().st_size > 0:
        return outpath

    with session.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        desc = f"Téléchargement {filename}"
        with open(outpath, "wb") as f, tqdm(
            total=total if total > 0 else None,
            unit="B", unit_scale=True, desc=desc
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1 MiB
                if chunk:
                    f.write(chunk)
                    if total > 0:
                        pbar.update(len(chunk))
    return outpath


# --- Lecture & plot par fichier (faible RAM) ---------------------------------------------

def lon_to_range(lon: np.ndarray, target: str = "-180_180") -> np.ndarray:
    if target == "-180_180":
        lon2 = ((lon + 180) % 360) - 180
    elif target == "0_360":
        lon2 = lon % 360
    else:
        raise ValueError("target must be '-180_180' or '0_360'")
    return lon2


def read_and_plot_ssha(
    nc_path: Path,
    ax: plt.Axes,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    prefer_var: str = "ssha_filtered",
    fallback_var: str = "ssha",
    alpha: float = 0.9,
    vmin: float = None,
    vmax: float = None,
    cmap=None,
    norm=None
) -> tuple[bool, object]:
    """
    Open a netcdf, read (lon, lat, SSHa), filter on Med
    """
    with Dataset(nc_path, "r") as ds:
        # Détection noms de variables
        var_lon = "longitude" if "longitude" in ds.variables else "lon"
        var_lat = "latitude" if "latitude" in ds.variables else "lat"
        if prefer_var in ds.variables:
            var_ssha = prefer_var
        elif fallback_var in ds.variables:
            var_ssha = fallback_var
        else:
            print(f"[WARN] {nc_path.name}: variables SSHa introuvables ({prefer_var}/{fallback_var}).")
            return False

        lon = ds.variables[var_lon][:]
        lat = ds.variables[var_lat][:]
        ssha = ds.variables[var_ssha][:]

        # Gestion des FILL/MASK
        if hasattr(ds.variables[var_ssha], "_FillValue"):
            fv = ds.variables[var_ssha]._FillValue
            ssha = np.ma.masked_equal(ssha, fv)
        if hasattr(ds.variables[var_ssha], "missing_value"):
            mv = ds.variables[var_ssha].missing_value
            ssha = np.ma.masked_equal(ssha, mv)

        lon = lon_to_range(np.array(lon), target="-180_180")

        def in_box(lonv, latv):
            return (latv >= lat_min) & (latv <= lat_max) & (lonv >= lon_min) & (lonv <= lon_max)

        plotted = False
        if ssha.ndim == 2:
            mask = in_box(lon, lat)
            if mask.any():
                ssha2 = np.ma.array(ssha, mask=~mask)
                # pcolormesh attend des grilles 2D alignées
            pcm = ax.pcolormesh(
                lon, lat, ssha2, shading="auto", alpha=alpha,
                transform=ccrs.PlateCarree(),
                cmap=cmap, norm=norm
            )
            return True, pcm            
        elif ssha.ndim == 1:
            mask = in_box(lon, lat)
            if mask.any():
                sc = ax.scatter(
                    lon[mask], lat[mask], c=ssha[mask], s=1, alpha=alpha,
                    transform=ccrs.PlateCarree(),
                    cmap=cmap, norm=norm
                )
                return True, sc                
        else:
            print(f"[WARN] {nc_path.name}: dimension SSHa {ssha.shape} non gérée.")
            return False, None

        return plotted


# --- Main -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Télécharger et cartographier SWOT L3 Expert pour un cycle.")
    parser.add_argument("--cycle", type=int, required=True, help="Numéro de cycle (ex: 2)")
    parser.add_argument("--passes-file", type=str, default=None,
                        help="Fichier texte listant les passes (ex: passes_region.txt avec 001 par ligne)")
    parser.add_argument("--lat-min", type=float, default=30.0)
    parser.add_argument("--lat-max", type=float, default=46.0)
    parser.add_argument("--lon-min", type=float, default=-6.0)
    parser.add_argument("--lon-max", type=float, default=36.0)
    parser.add_argument("--outdir", type=str, default="swot_l3_downloads",
                        help="Répertoire local où stocker les .nc")
    parser.add_argument("--username", type=str, default=None)
    parser.add_argument("--password", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limiter le nombre de fichiers (debug)")
    parser.add_argument("--show", action="store_true", help="Afficher la figure à la fin")
    parser.add_argument("--savefig", type=str, default=None, help="Chemin de sauvegarde de la figure PNG")
    parser.add_argument("--vmin", type=float, default=None,
                        help="Valeur SSHa minimale pour la colorbar commune (m)")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Valeur SSHa maximale pour la colorbar commune (m)")
    parser.add_argument("--cmap", type=str, default="viridis",
                        help="Nom du colormap Matplotlib à utiliser (ex: RdBu_r, coolwarm, viridis)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    session = build_session(args.username, args.password)

    print(f"→ Parcours du catalogue cycle_{args.cycle:03d} ...")
    datasets = list_cycle_datasets(args.cycle, session)

    if not datasets:
        print("Aucun dataset trouvé. Vérifie le cycle et l'accès au catalogue.")
        sys.exit(1)

    # Filtrage sur les passes si demandé
    wanted_passes: Optional[Set[int]] = None
    if args.passes_file:
        txt = Path(args.passes_file).read_text().strip().splitlines()
        wanted_passes = {int(x) for x in txt if x.strip().isdigit()}
        print(f"→ Passes demandées: {sorted(wanted_passes)}")

    selected: List[Tuple[str, str]] = []
    
    for name, url_path in datasets:
        cyc, pas = parse_cycle_pass_from_name(name)
        if cyc is None or pas is None:
            continue
        if wanted_passes is not None and pas not in wanted_passes:
            continue
        selected.append((name, url_path))

    if args.limit:
        selected = selected[: args.limit]

    if not selected:
        print("Aucun fichier correspondant aux passes/au cycle fournis.")
        sys.exit(0)

    print(f"→ {len(selected)} fichiers à traiter.")

    # Figure Med

    fig = plt.figure(figsize=(9, 6), constrained_layout=True)
    ax = plt.axes(projection=ccrs.PlateCarree())
    

    land = cfeature.NaturalEarthFeature(
        "physical", "land", "50m", edgecolor="black", facecolor="lightgray"
    )
    ax.add_feature(land, zorder=0)
    ax.coastlines(resolution="50m", color="black", linewidth=0.6)
    
    # Graticules / grille
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    
    ax.set_title(f"SWOT L3 Expert SSHa — cycle {args.cycle:03d}")
    

    ax.set_extent([args.lon_min, args.lon_max, args.lat_min, args.lat_max], crs=ccrs.PlateCarree())

    
    # --- Option : calcul auto de vmin/vmax si non fournis ---
    auto_vmin, auto_vmax = args.vmin, args.vmax
    if auto_vmin is None or auto_vmax is None:
        import numpy as np
        vals = []
        for _, url_path in selected[:min(len(selected), 20)]:  # limite à 20 fichiers pour aller vite
            local_nc = (Path(args.outdir) / Path(url_path).name
                        if Path(args.outdir).joinpath(Path(url_path).name).exists()
                        else None)
            if local_nc and local_nc.exists():
                from netCDF4 import Dataset
                with Dataset(local_nc, "r") as ds:
                    var = ds.variables.get("ssha_filtered") or ds.variables.get("ssha")
                    if var is not None:
                        data = var[:].compressed() if hasattr(var[:], "compressed") else var[:]
                        finite = np.isfinite(data)
                        if finite.any():
                            vals.extend(data[finite].ravel())
        if vals:
            arr = np.array(vals)
            auto_vmin, auto_vmax = np.percentile(arr, [2, 98])  # plage robuste à 2–98 %
            print(f"[INFO] Échelle SSHa estimée auto: vmin={auto_vmin:.3f}, vmax={auto_vmax:.3f}")

    norm = colors.Normalize(vmin=auto_vmin, vmax=auto_vmax)
    cmap = cm.get_cmap(args.cmap)
    last_mappable = None
    n_plotted = 0
    for name, url_path in selected:
        local_nc = download_fileserver_file(url_path, outdir, session)
        ok, mappable = read_and_plot_ssha(
            local_nc, ax,
            lat_min=args.lat_min, lat_max=args.lat_max,
            lon_min=args.lon_min, lon_max=args.lon_max,
            prefer_var="ssha_filtered", fallback_var="ssha", alpha=0.9,
            vmin=auto_vmin, vmax=auto_vmax,
            cmap=cmap, norm=norm
        )
        if ok:
            cyc, pas = parse_cycle_pass_from_name(name)
            n_plotted += 1
            if mappable is not None:
                last_mappable = mappable

    print(f"→ Tracé terminé : {n_plotted} fichier(s) représenté(s).")

    # --- Étape 5 : colorbar commune ---
    if last_mappable is not None:
        cbar = fig.colorbar(last_mappable, ax=ax, orientation="vertical", shrink=0.5, pad=0.02)
        cbar.set_label("SSHa (m)")
    else:
        print("[WARN] Aucun mappable pour la colorbar (rien n'a été tracé).")

    if args.savefig:
        outpng = Path(args.savefig)
        outpng.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpng, dpi=180)
        print(f"Figure sauvegardée : {outpng}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()


