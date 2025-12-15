"""
Module: fsr_features.py
Purpose: Compute per-sample and per-repetition features from 4x4 FSR (force-sensing resistor) insole data,
         including regional load fractions (heel/fore, medial/lateral), instantaneous ratios, center of pressure (CoP),
         activation percentage, and robust summary statistics. Also includes utilities to combine left/right feet.

Main functionalities
- Discover FSR columns in a DataFrame and map them to a 4x4 grid (row-major order).
- Compute per-sample features (time-indexed): regional sums/fractions, ratios, CoP, activation %, spatial dispersion.
- Aggregate per-repetition features: legacy means, robust stats for CoP and ratios, CoP path length and path speed.
- Combine left/right repetition-level features into symmetric "both" metrics and left-right asymmetries.

Expected input columns (per-sample DataFrame)
- 'ReconstructedTime' : float-like timestamp
- 'label'             : activity label for the sample
- 'rep_id'            : repetition identifier
- 'Fsr.01'..'Fsr.16'  : FSR channels (exact casing and numeric suffix expected)

Outputs
- Per-sample features: DataFrame with additional columns (suffix _{side}, where side in {'L','R'}).
- Per-repetition features: one row per (rep_id, label) with summary stats and robust descriptors.
- Combined L/R features: "both" averages and L/R asymmetry metrics.

Internal dependencies in the project
- None beyond standard Python, NumPy, and pandas.

Notes
- The grid is assumed 4x4 in row-major order after sorting channel names by their numeric suffix (1..16).
- If your physical mapping differs (e.g., heel/fore inverted), swap the masks inside _region_masks accordingly.
- No changes to the original logic; only English docstrings/comments standardization.
"""

from __future__ import annotations

import re
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd


# =====================================
# Column discovery & 4x4 grid utilities
# =====================================
def fsr_columns(df: pd.DataFrame) -> List[str]:
    """
    Return the list of FSR column names, sorted by their numeric index.

    Expected column naming convention: 'Fsr.01'..'Fsr.16'. The final order is ascending by numeric index.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing FSR channels among its columns.

    Returns
    -------
    list of str
        Sorted list of FSR channel names.
    """
    cols = [c for c in df.columns if c.startswith("Fsr.")]

    def _num(c: str) -> int:
        m = re.search(r"(\d+)", c)
        return int(m.group(1)) if m else 10**9

    return sorted(cols, key=_num)


def default_grid_coords() -> Dict[str, Tuple[float, float]]:
    """
    Coordinate normalizzate (x,y) in [0,1] per la soletta Yeti 42/43,
    coerenti con il layout fisico confermato:

        r0: [13 11 12 14]  (tallone)
        r1: [08 10 15 05]
        r2: [07 09 16 04]
        r3: [06 01 02 03]  (avampiede/dita)

    x = col/(3)  mediale(0) -> laterale(3)
    y = row/(3)  posteriore(0) -> anteriore(3)
    """
    # mappa (row, col) per ciascun canale Fsr.xx
    rc_map = {
        "Fsr.13": (0, 0), "Fsr.11": (0, 1), "Fsr.12": (0, 2), "Fsr.14": (0, 3),
        "Fsr.08": (1, 0), "Fsr.10": (1, 1), "Fsr.15": (1, 2), "Fsr.05": (1, 3),
        "Fsr.07": (2, 0), "Fsr.09": (2, 1), "Fsr.16": (2, 2), "Fsr.04": (2, 3),
        "Fsr.06": (3, 0), "Fsr.01": (3, 1), "Fsr.02": (3, 2), "Fsr.03": (3, 3),
    }

    coords: Dict[str, Tuple[float, float]] = {}
    for ch, (r, c) in rc_map.items():
        x = c / 3.0  # mediale->laterale
        y = r / 3.0  # posteriore->anteriore
        coords[ch] = (x, y)
    return coords



def _region_masks(cols: List[str]) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Indici (rispetto a `cols`) per regioni: heel, fore, medial, lateral
    costruiti sulla base del layout fisico Yeti 42/43 (vedi default_grid_coords).
    """
    # stessa mappa (row,col) usata sopra
    rc_map = {
        "Fsr.13": (0, 0), "Fsr.11": (0, 1), "Fsr.12": (0, 2), "Fsr.14": (0, 3),
        "Fsr.08": (1, 0), "Fsr.10": (1, 1), "Fsr.15": (1, 2), "Fsr.05": (1, 3),
        "Fsr.07": (2, 0), "Fsr.09": (2, 1), "Fsr.16": (2, 2), "Fsr.04": (2, 3),
        "Fsr.06": (3, 0), "Fsr.01": (3, 1), "Fsr.02": (3, 2), "Fsr.03": (3, 3),
    }

    # costruisco lista (idx_in_cols, row, col) per i canali presenti
    info: List[Tuple[int, int, int]] = []
    for i, ch in enumerate(cols):
        if ch not in rc_map:
            raise KeyError(f"Column '{ch}' non trovata nel mapping fisico (atteso Fsr.01..Fsr.16).")
        r, c = rc_map[ch]
        info.append((i, r, c))

    # regioni (righe 0-1 = heel/posteriore; righe 2-3 = fore/anteriore)
    heel    = [i for i, r, c in info if r in (0, 1)]
    fore    = [i for i, r, c in info if r in (2, 3)]
    medial  = [i for i, r, c in info if c in (0, 1)]
    lateral = [i for i, r, c in info if c in (2, 3)]
    return heel, fore, medial, lateral



# =============================
# Small statistical helpers
# =============================
def _iqr(arr: np.ndarray) -> float:
    """Return the interquartile range (Q3 - Q1) of a 1D array."""
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


def _cv(arr: np.ndarray, eps: float = 1e-9) -> float:
    """Return the coefficient of variation (std / mean) with a small epsilon in the denominator."""
    m = float(np.mean(arr))
    s = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return float(s / (m + eps))


# ==================================
# Per-sample (time-stamped) features
# ==================================
def per_sample_features(
    df: pd.DataFrame,
    side: str,
    coords: Optional[Dict[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """
    Compute time-indexed features per sample: regional sums/fractions, per-sample ratios, CoP, activation %, dispersion.

    Requirements on `df`
    --------------------
    - Columns: 'Fsr.01'..'Fsr.16' (float-like), 'ReconstructedTime', 'label', 'rep_id'.

    Parameters
    ----------
    df : pandas.DataFrame
        Input raw data.
    side : str
        Side suffix to append to output columns (e.g., 'L' or 'R').
    coords : dict[str, tuple[float, float]], optional
        Mapping from channel name to (x, y) coordinates in [0,1]. If None, use default_grid_coords().

    Returns
    -------
    pandas.DataFrame
        Per-sample feature DataFrame; retains 'ReconstructedTime', 'label', 'rep_id' from input.
    """
    if coords is None:
        coords = default_grid_coords()

    cols = fsr_columns(df)                      # 'Fsr.01'..'Fsr.16'
    X = df[cols].astype(float).to_numpy()       # shape (T, 16)
    eps = 1e-9

    # Region masks (computed once)
    H, F, M, L = _region_masks(cols)

    # Regional absolute sums
    heel_sum    = X[:, H].sum(axis=1)
    fore_sum    = X[:, F].sum(axis=1)
    medial_sum  = X[:, M].sum(axis=1)
    lateral_sum = X[:, L].sum(axis=1)
    total_sum   = X.sum(axis=1)

    # Regional fractions relative to the total
    fore_frac    = fore_sum   / (total_sum + eps)
    heel_frac    = heel_sum   / (total_sum + eps)
    medial_frac  = medial_sum / (total_sum + eps)
    lateral_frac = lateral_sum/ (total_sum + eps)

    # Robust per-sample ratios
    foreheel_ratio = fore_sum   / (heel_sum + eps)
    medlat_ratio   = medial_sum / (lateral_sum + eps)

    # 2D center of pressure (weighted by FSR values)
    xy   = np.array([coords[c] for c in cols], dtype=float)  # (16,2)
    cop  = (X @ xy) / (total_sum.reshape(-1, 1) + eps)
    cop_x, cop_y = cop[:, 0], cop[:, 1]

    # Activation percentage (threshold relative to global distribution)
    thr = np.percentile(X, 95) * 0.10
    active_fraction = (X >= thr).sum(axis=1) / X.shape[1]

    # Instantaneous spatial dispersion across sensors
    spatial_std = X.std(axis=1, ddof=0)
    spatial_var = spatial_std ** 2

    out = pd.DataFrame({
        #"ReconstructedTime": df["ReconstructedTime"].values,
        #"label": df["label"].values,
        #"rep_id": df["rep_id"].values,

        # Load / activation
        f"total_sum_{side}": total_sum,
        f"avg_press_{side}": X.mean(axis=1),
        f"active_fraction_{side}": active_fraction,

        # CoP
        f"cop_x_{side}": cop_x,
        f"cop_y_{side}": cop_y,

        # Dispersion
        f"spatial_std_{side}": spatial_std,
        f"spatial_var_{side}": spatial_var,

        # Regional fractions
        f"fore_frac_{side}":    fore_frac,
        f"heel_frac_{side}":    heel_frac,
        f"medial_frac_{side}":  medial_frac,
        f"lateral_frac_{side}": lateral_frac,

        # Per-sample ratios
        f"foreheel_ratio_{side}": foreheel_ratio,
        f"medlat_ratio_{side}":   medlat_ratio,
    })

    # Lightweight DEBUG (first rows)
    if len(out) > 0:
        print(
            f"[DEBUG per_sample {side}] "
            f"fore~{out[f'fore_frac_{side}'].head(3).round(3).tolist()} "
            f"heel~{out[f'heel_frac_{side}'].head(3).round(3).tolist()} "
            f"med~{out[f'medial_frac_{side}'].head(3).round(3).tolist()} "
            f"lat~{out[f'lateral_frac_{side}'].head(3).round(3).tolist()}"
        )
    return out


# ====================================
# Per-repetition (rep_id) aggregation
# ====================================
def _cop_path_len(x: np.ndarray, y: np.ndarray) -> float:
    """Return the polyline length of the CoP path."""
    dx = np.diff(x)
    dy = np.diff(y)
    return float(np.sum(np.sqrt(dx * dx + dy * dy)))


def aggregate_per_rep(per_sample: pd.DataFrame, side: str) -> pd.DataFrame:
    """
    Aggregate features per repetition (rep_id).

    Keeps backward-compatible fields (avg_* etc.) and adds robust statistics for CoP and ratios:
    means / std / IQR / CV, along with CoP path length and path length per second.

    Parameters
    ----------
    per_sample : pandas.DataFrame
        Per-sample features as returned by `per_sample_features`.
    side : str
        Side suffix used in per-sample features (e.g., 'L' or 'R').

    Returns
    -------
    pandas.DataFrame
        One row per (rep_id, label) with aggregated fields.
    """
    rows: List[Dict[str, float]] = []
    for (rep, label), g in per_sample.groupby(["rep_id", "label"]):
        # Load (legacy & summaries)
        total_mean = float(g[f"total_sum_{side}"].mean())

        avg_mean = float(g[f"avg_press_{side}"].mean())
        avg_std  = float(g[f"avg_press_{side}"].std(ddof=0))
        avg_var  = float(avg_std ** 2)

        spatial_std_mean = float(g[f"spatial_std_{side}"].mean())
        spatial_var_mean = float(g[f"spatial_var_{side}"].mean())

        activation_pct = float(g[f"active_fraction_{side}"].mean() * 100.0)

        # CoP robust stats
        cop_x = g[f"cop_x_{side}"].to_numpy()
        cop_y = g[f"cop_y_{side}"].to_numpy()
        cop_x_mean = float(cop_x.mean())
        cop_y_mean = float(cop_y.mean())
        cop_x_std  = float(cop_x.std(ddof=0))
        cop_y_std  = float(cop_y.std(ddof=0))
        cop_x_iqr  = _iqr(cop_x)
        cop_y_iqr  = _iqr(cop_y)
        cop_x_cv   = _cv(cop_x)
        cop_y_cv   = _cv(cop_y)
        path_len   = _cop_path_len(cop_x, cop_y)

        # Mean regional fractions (time-average)
        fore_mean    = float(g[f"fore_frac_{side}"].mean())
        heel_mean    = float(g[f"heel_frac_{side}"].mean())
        medial_mean  = float(g[f"medial_frac_{side}"].mean())
        lateral_mean = float(g[f"lateral_frac_{side}"].mean())

        # Ratios computed from means (backward compatibility)
        EPS = 1e-9
        foreheel_ratio_from_means = float(fore_mean / (heel_mean + EPS))
        medlat_ratio_from_means   = float(medial_mean / (lateral_mean + EPS))

        # Ratio statistics directly from per-sample series
        fr_series = g[f"foreheel_ratio_{side}"].to_numpy()
        ml_series = g[f"medlat_ratio_{side}"].to_numpy()

        fr_mean = float(fr_series.mean())
        fr_std  = float(fr_series.std(ddof=0))
        fr_iqr  = _iqr(fr_series)
        fr_cv   = _cv(fr_series)

        ml_mean = float(ml_series.mean())
        ml_std  = float(ml_series.std(ddof=0))
        ml_iqr  = _iqr(ml_series)
        ml_cv   = _cv(ml_series)

        # Duration and normalized CoP path
        t = g["ReconstructedTime"].to_numpy()
        duration = float(t[-1] - t[0]) if len(t) > 1 else 1.0
        path_per_sec = float(path_len / max(duration, 1e-9))

        rows.append({
            "rep_id": rep,
            "label": label,

            # Load
            f"total_mean_{side}": total_mean,

            # Legacy
            f"avg_mean_{side}": avg_mean,
            f"avg_std_{side}": avg_std,
            f"avg_var_{side}": avg_var,
            f"spatial_std_mean_{side}": spatial_std_mean,
            f"spatial_var_mean_{side}": spatial_var_mean,
            f"activation_pct_{side}": activation_pct,

            # CoP (robust stats)
            f"cop_x_mean_{side}": cop_x_mean,
            f"cop_y_mean_{side}": cop_y_mean,
            f"cop_x_std_{side}":  cop_x_std,
            f"cop_y_std_{side}":  cop_y_std,
            f"cop_x_iqr_{side}":  cop_x_iqr,
            f"cop_y_iqr_{side}":  cop_y_iqr,
            f"cop_x_cv_{side}":   cop_x_cv,
            f"cop_y_cv_{side}":   cop_y_cv,

            f"cop_path_len_{side}": path_len,
            f"cop_path_per_sec_{side}": path_per_sec,
            f"samples_{side}": int(len(g)),

            # Ratios from means (compat)
            f"foreheel_ratio_{side}": foreheel_ratio_from_means,
            f"medlat_ratio_{side}":   medlat_ratio_from_means,

            # Ratio stats from per-sample
            f"foreheel_ratio__mean_{side}": fr_mean,
            f"foreheel_ratio__std_{side}":  fr_std,
            f"foreheel_ratio__iqr_{side}":  fr_iqr,
            f"foreheel_ratio__cv_{side}":   fr_cv,

            f"medlat_ratio__mean_{side}": ml_mean,
            f"medlat_ratio__std_{side}":  ml_std,
            f"medlat_ratio__iqr_{side}":  ml_iqr,
            f"medlat_ratio__cv_{side}":   ml_cv,
        })
    return pd.DataFrame(rows)


# ===============================
# Left/Right repetition merging
# ===============================
def combine_left_right(rep_L: pd.DataFrame, rep_R: pd.DataFrame) -> pd.DataFrame:
    """
    Merge left and right repetition-level features on (rep_id, label), compute "both" averages,
    and left-right asymmetry metrics (diff/absdiff) for selected indicators.

    Parameters
    ----------
    rep_L : pandas.DataFrame
        Repetition-level features for the left foot (columns suffixed with '_L').
    rep_R : pandas.DataFrame
        Repetition-level features for the right foot (columns suffixed with '_R').

    Returns
    -------
    pandas.DataFrame
        Combined features per (rep_id, label) with:
        - <base>_both: mean of left/right for selected bases
        - load_asym: normalized load asymmetry using total_mean_L/R
        - <base>_diff / <base>_absdiff: signed and absolute differences for selected bases
    """
    df = pd.merge(rep_L, rep_R, on=["rep_id", "label"], how="inner")
    out = pd.DataFrame({"rep_id": df["rep_id"], "label": df["label"]})

    def pair_mean(base: str) -> np.ndarray:
        return df[[f"{base}_L", f"{base}_R"]].mean(axis=1).to_numpy()

    def pair_diff(base: str) -> np.ndarray:
        return (df[f"{base}_R"] - df[f"{base}_L"]).to_numpy()

    # --- "Both" means (keep the most informative bases) ---
    for base in [
        "avg_mean", "avg_std", "avg_var",
        "spatial_std_mean", "spatial_var_mean",
        "activation_pct",
        "cop_x_mean", "cop_y_mean",
        "cop_x_std", "cop_y_std",
        "cop_x_iqr", "cop_y_iqr",
        "cop_x_cv",  "cop_y_cv",
        "cop_path_len", "cop_path_per_sec",
        "samples",
        "foreheel_ratio", "medlat_ratio",
        "foreheel_ratio__mean", "foreheel_ratio__std", "foreheel_ratio__iqr", "foreheel_ratio__cv",
        "medlat_ratio__mean",   "medlat_ratio__std",   "medlat_ratio__iqr",   "medlat_ratio__cv",
        "total_mean",
    ]:
        cols = [f"{base}_L", f"{base}_R"]
        if all(c in df.columns for c in cols):
            out[f"{base}_both"] = pair_mean(base)

    # --- Left/Right asymmetries (diff and absdiff) ---
    eps = 1e-9

    # Load asymmetry using total_mean as a proxy for foot loading
    if {"total_mean_L", "total_mean_R"}.issubset(df.columns):
        load_asym = (df["total_mean_L"] - df["total_mean_R"]) / (df["total_mean_L"] + df["total_mean_R"] + eps)
        out["load_asym"] = load_asym.to_numpy()

    # Differences for ratios and CoP coordinates
    for base in ["foreheel_ratio", "medlat_ratio", "cop_x_mean", "cop_y_mean"]:
        cols = [f"{base}_L", f"{base}_R"]
        if all(c in df.columns for c in cols):
            d = pair_diff(base)
            out[f"{base}_diff"]    = d
            out[f"{base}_absdiff"] = np.abs(d)

    return out


def aggregate_per_window(per_sample: pd.DataFrame, side: str,
                         start_idx: int, stop_idx: int):
    """
    
    THIS IS A BIT COUNTERINTUITIVE AS IT STANDS NOW, BUT IT DOES MAKE SENSE. PASS ONLY A SIGLE WINDOW INTO THIS.
    start_idx and stop_idx do nothing at the moment, and that is CORRECT AS IMPLEMENTATION IS NOW. fix later
    
    Aggregate features for a single time window [start_idx:stop_idx) in per_sample.

    Returns a one-row DataFrame with (rep_id, label, window_id) and the same
    metrics you compute in aggregate_per_rep, but restricted to the slice.
    """
    g = per_sample.copy()
    if g.empty:
        return pd.DataFrame([])

    # sort by time to get correct duration/path stats
    # if "ReconstructedTime" in g:
    #     g = g.sort_values("ReconstructedTime")
    

    # rep_id/label for this window
    rep = g["rep_id"].iloc[0] if "rep_id" in g else None
    # majority/first label – pick what fits your data
    if "label" in g:
        label = g["label"].mode().iat[0] if not g["label"].mode().empty else g["label"].iloc[0]
    else:
        label = None

    # Basic loads/dispersion (same fields as your per-rep)
    total_mean        = float(g[f"total_sum_{side}"].mean())

    avg_mean          = float(g[f"avg_press_{side}"].mean())
    avg_std           = float(g[f"avg_press_{side}"].std(ddof=0))
    avg_var           = float(avg_std ** 2)

    spatial_std_mean  = float(g[f"spatial_std_{side}"].mean())
    spatial_var_mean  = float(g[f"spatial_var_{side}"].mean())

    activation_pct    = float(g[f"active_fraction_{side}"].mean() * 100.0)

    # CoP stats
    cop_x = g[f"cop_x_{side}"].to_numpy()
    cop_y = g[f"cop_y_{side}"].to_numpy()
    cop_x_mean = float(cop_x.mean());  cop_y_mean = float(cop_y.mean())
    cop_x_std  = float(cop_x.std(ddof=0));  cop_y_std  = float(cop_y.std(ddof=0))
    cop_x_iqr  = _iqr(cop_x);  cop_y_iqr  = _iqr(cop_y)
    cop_x_cv   = _cv(cop_x);   cop_y_cv   = _cv(cop_y)
    path_len   = _cop_path_len(cop_x, cop_y)

    # Duration + normalized CoP path
    if "ReconstructedTime" in g:
        t = g["ReconstructedTime"].to_numpy()
        duration = float(t[-1] - t[0]) if len(t) > 1 else 1.0
    else:
        duration = float(len(g))  # fallback to samples
    path_per_sec = float(path_len / max(duration, 1e-9))

    # Regional fractions
    fore_mean    = float(g[f"fore_frac_{side}"].mean())
    heel_mean    = float(g[f"heel_frac_{side}"].mean())
    medial_mean  = float(g[f"medial_frac_{side}"].mean())
    lateral_mean = float(g[f"lateral_frac_{side}"].mean())

    EPS = 1e-9
    foreheel_ratio_from_means = float(fore_mean / (heel_mean + EPS))
    medlat_ratio_from_means   = float(medial_mean / (lateral_mean + EPS))

    # Ratio series stats
    fr_series = g[f"foreheel_ratio_{side}"].to_numpy()
    ml_series = g[f"medlat_ratio_{side}"].to_numpy()

    fr_mean = float(fr_series.mean()); fr_std = float(fr_series.std(ddof=0))
    fr_iqr  = _iqr(fr_series);         fr_cv  = _cv(fr_series)

    ml_mean = float(ml_series.mean()); ml_std = float(ml_series.std(ddof=0))
    ml_iqr  = _iqr(ml_series);         ml_cv  = _cv(ml_series)

    row = {
        # Load / legacy
        f"total_mean_{side}": total_mean,
        f"avg_mean_{side}": avg_mean,
        f"avg_std_{side}":  avg_std,
        f"avg_var_{side}":  avg_var,
        f"spatial_std_mean_{side}": spatial_std_mean,
        f"spatial_var_mean_{side}": spatial_var_mean,
        f"activation_pct_{side}": activation_pct,

        # CoP stats
        f"cop_x_mean_{side}": cop_x_mean,  f"cop_y_mean_{side}": cop_y_mean,
        f"cop_x_std_{side}":  cop_x_std,   f"cop_y_std_{side}":  cop_y_std,
        f"cop_x_iqr_{side}":  cop_x_iqr,   f"cop_y_iqr_{side}":  cop_y_iqr,
        f"cop_x_cv_{side}":   cop_x_cv,    f"cop_y_cv_{side}":   cop_y_cv,
        f"cop_path_len_{side}":       path_len,
        f"cop_path_per_sec_{side}":   path_per_sec,
        #f"samples_{side}": int(len(g)),

        # Ratios from means
        f"foreheel_ratio_{side}": foreheel_ratio_from_means,
        f"medlat_ratio_{side}":   medlat_ratio_from_means,

        # Ratio series stats
        f"foreheel_ratio__mean_{side}": fr_mean,
        f"foreheel_ratio__std_{side}":  fr_std,
        f"foreheel_ratio__iqr_{side}":  fr_iqr,
        f"foreheel_ratio__cv_{side}":   fr_cv,

        f"medlat_ratio__mean_{side}": ml_mean,
        f"medlat_ratio__std_{side}":  ml_std,
        f"medlat_ratio__iqr_{side}":  ml_iqr,
        f"medlat_ratio__cv_{side}":   ml_cv,
    }
    return row


def aggregate_over_windows(per_sample: pd.DataFrame,
                           side: str,
                           window_size: int,
                           step_size: int | None = None,
                           time_col: str = "ReconstructedTime") -> pd.DataFrame:
    """
    Iterate over per_sample in windows of fixed length (number of samples).
    
    Args:
        per_sample : DataFrame with per-sample features.
        side       : "L" or "R".
        window_size: number of samples per window.
        step_size  : how far to move the window each time (default = window_size → non-overlapping).
        time_col   : column to use for ordering (default = "ReconstructedTime").
    
    Returns:
        DataFrame with one row per window, same fields as aggregate_per_window, plus window_id.
    """
    if step_size is None:
        step_size = window_size

    # Ensure sorted by time if available
    if time_col in per_sample.columns:
        per_sample = per_sample.sort_values(time_col).reset_index(drop=True)
    else:
        per_sample = per_sample.reset_index(drop=True)

    rows = []
    n = len(per_sample)
    window_id = 0
    for start in range(0, n, step_size):
        stop = start + window_size
        if stop > n:   # drop incomplete last window
            break
        window_df = aggregate_per_window(per_sample, side, start, stop)
        if not window_df.empty:
            window_df["window_id"] = window_id
            rows.append(window_df)
            window_id += 1

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()



