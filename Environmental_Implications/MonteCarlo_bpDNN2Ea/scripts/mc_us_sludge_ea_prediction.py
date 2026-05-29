"""
Monte Carlo Prediction of Activation Energy (Ea) for U.S. Sewage Sludge
========================================================================
This script mirrors the functionality of *mc_us_sludge_prediction.py* (pyrolysis
products) but targets the **Activation Energy (Ea)** neural-network model in the
*bpDNN2Ea_AshOptimized* project.

Workflow
--------
1.  Load the trained MATLAB neural-network (``Results_trained.mat``).
2.  Extract the original training data *X_train* / *y_train* and feature names.
3.  Generate a Monte-Carlo bootstrap sample of the input space by sampling
    **with replacement** from the training data (user-configurable *n*).
    •  This guarantees physically plausible combinations without hand-crafted
       mass-balance constraints.
4.  Apply the **same min-max scaling** that was used during MATLAB training
    (see ``net.processFcn = 'mapminmax'`` in *bpDNN4Ea.m*).
5.  Predict Ea for each Monte-Carlo sample.
6.  Inversely transform network outputs back to absolute units (kJ/mol).
7.  Save:
    •  ``mc_ea_predictions.csv`` – raw predictions + sampled features.
    •  ``mc_ea_distribution.png`` / ``.svg`` – violin + histogram of Ea.

Usage:
---
    python mc_us_sludge_ea_prediction.py --samples 10000 --seed 123

"""

from __future__ import annotations

# Set threading environment variables before importing numpy to prevent OpenBLAS/MKL thread thrashing
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import sys
import multiprocessing
import warnings
from pathlib import Path
from typing import Tuple

# Suppress annoying SciPy warning about duplicate variable name 'None' in matlab files
warnings.filterwarnings("ignore", message="Duplicate variable name")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# Add project root so we can import helper utilities from shap_analysis_ea.py
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Import helper utilities defined in shap_analysis_ea.py (same folder level)
from shap_analysis_ea import (  # type: ignore
    load_matlab_data,
    extract_neural_network_data,
    generate_feature_names,
    MatlabNeuralNetworkWrapper,
)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def load_trained_model(mat_path: Path) -> Tuple[
    MatlabNeuralNetworkWrapper,
    list[str],
    np.ndarray,
    np.ndarray,
]:
    """Return (wrapper, feature_names, X_train, y_train)."""
    print(f"Loading MATLAB model from: {mat_path}")
    mat_data = load_matlab_data(str(mat_path))

    X_train, y_train, net_struct = extract_neural_network_data(mat_data)
    feature_names = generate_feature_names(X_train, mat_data)

    wrapper = MatlabNeuralNetworkWrapper(net_struct)
    print("Model wrapper initialised – ready for prediction.")
    return wrapper, feature_names, X_train, y_train


def align_sample_features(df_samples: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """Ensure DataFrame columns are in the exact order expected by the model."""
    aligned_dict = {}
    for feat in feature_names:
        if feat in df_samples.columns:
            aligned_dict[feat] = df_samples[feat].values
        else:
            aligned_dict[feat] = np.zeros(len(df_samples))
    aligned = pd.DataFrame(aligned_dict, index=df_samples.index)
    return aligned.values


def plot_ea_distribution(ea_kjmol: pd.Series, output_path: Path) -> None:
    """Save violin + histogram plot to *output_path* (PNG & SVG)."""
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))

    sns.violinplot(y=ea_kjmol, inner="quartile", color="skyblue", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Activation Energy Ea (kJ/mol)")
    ax.set_title("Monte Carlo Prediction Uncertainty – Ea of U.S. Sewage Sludge")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".svg"), format="svg")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Additional helper utilities borrowed and adapted from mc_us_sludge_prediction.py
# -----------------------------------------------------------------------------

import tempfile
import shutil

# Mapping of alternative column names to canonical model feature names
CANONICAL_NAME_MAP = {
    "volatilematter": "VolatileMatters/%",
    "volatilematters": "VolatileMatters/%",
    "vm": "VolatileMatters/%",
    "fixedcarbon": "FixedCarbon/%",
    "fc": "FixedCarbon/%",
    "ash": "Ash/%",
    "c": "C/%",
    "h": "H/%",
    "o": "O/%",
    "n": "N/%",
    "s": "S/%",
    "sio2": "Ash_SiO2",
    "na2o": "Ash_Na2O",
    "mgo": "Ash_MgO",
    "al2o3": "Ash_Al2O3",
    "k2o": "Ash_K2O",
    "cao": "Ash_CaO",
    "p2o5": "Ash_P2O5",
    "cuo": "Ash_CuO",
    "zno": "Ash_ZnO",
    "fe2o3": "Ash_Fe2O3",
    "degreeconversion": "Degree_conversion",
}


def _clean_key(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _canonical_no_space(name: str) -> str:
    """Return cleaned key (alphanumerics only)."""
    return _clean_key(name)


INT_SAMPLED_FEATURES: set[str] = {"reactortype"}


def canonicalize_feature_name(name: str):
    """Return canonical model feature name given raw spreadsheet header."""
    if not isinstance(name, str):
        return name
    key = _clean_key(name)

    # Direct dictionary lookup first
    if key in CANONICAL_NAME_MAP:
        return CANONICAL_NAME_MAP[key]

    # Heuristic matching for oxides and special cases
    for oxide, std_name in [
        ("sio2", "Ash_SiO2"),
        ("na2o", "Ash_Na2O"),
        ("mgo", "Ash_MgO"),
        ("al2o3", "Ash_Al2O3"),
        ("k2o", "Ash_K2O"),
        ("cao", "Ash_CaO"),
        ("p2o5", "Ash_P2O5"),
        ("cuo", "Ash_CuO"),
        ("zno", "Ash_ZnO"),
        ("fe2o3", "Ash_Fe2O3"),
    ]:
        if oxide in key:
            return std_name

    if "degreeconversion" in key or "degreeofconversion" in key or "degconversion" in key:
        return "Degree_conversion"

    return name


def read_parameter_ranges(range_path: Path) -> pd.DataFrame:
    """Read Excel file containing min/max (or single mean) for each feature."""
    df_raw = pd.read_excel(range_path, header=0)
    df = df_raw.copy()

    def _find_col(keywords):
        for col in df.columns:
            cl = str(col).lower()
            if any(k in cl for k in keywords):
                return col
        return None

    min_col = _find_col(["min", "minimum", "lower", "low"])
    max_col = _find_col(["max", "maximum", "upper", "high"])
    feat_col = df.columns[0]

    if min_col and max_col:
        ranges = df[[feat_col, min_col, max_col]].copy()
        ranges.columns = ["feature", "min", "max"]
        ranges["min"] = pd.to_numeric(ranges["min"], errors="coerce")
        ranges["max"] = pd.to_numeric(ranges["max"], errors="coerce")
        ranges.set_index("feature", inplace=True)
    else:
        first_col_lower = df.iloc[:, 0].astype(str).str.lower()
        if (first_col_lower == "min").any() and (first_col_lower == "max").any():
            min_row = df[first_col_lower == "min"].iloc[0]
            max_row = df[first_col_lower == "max"].iloc[0]
            features = df.columns[1:]
            ranges = pd.DataFrame({
                "min": pd.to_numeric(min_row[1:], errors="coerce"),
                "max": pd.to_numeric(max_row[1:], errors="coerce"),
            }, index=features)
        else:
            numeric_df = df.select_dtypes(include=[np.number])
            ranges = pd.DataFrame({
                "min": numeric_df.min(),
                "max": numeric_df.max(),
            })

    ranges.index = [canonicalize_feature_name(idx) for idx in ranges.index]
    ranges = ranges[~ranges.index.duplicated(keep="first")]
    return ranges


def load_constant_features(constant_path: Path) -> pd.Series:
    df = pd.read_excel(constant_path, header=0)
    const_series = df.iloc[0] if len(df) == 1 else df.mean()
    const_series.index = [canonicalize_feature_name(idx) for idx in const_series.index]
    const_series = const_series[~const_series.index.duplicated(keep="first")]
    return const_series


def expand_single_point_ranges(ranges: pd.DataFrame, rel_variation: float = 0.10) -> None:
    """Modify *ranges* in-place: if min==max add ±rel_variation jitter."""
    for feat in ranges.index:
        lo, hi = ranges.at[feat, "min"], ranges.at[feat, "max"]
        if pd.isna(lo) and not pd.isna(hi):
            lo = hi
        if pd.isna(hi) and not pd.isna(lo):
            hi = lo
        if pd.isna(lo) or pd.isna(hi):
            continue
        if np.isclose(lo, hi):
            mid = lo
            if mid != 0 and np.isfinite(mid):
                delta = abs(mid) * rel_variation
                ranges.at[feat, "min"] = mid - delta
                ranges.at[feat, "max"] = mid + delta


# Feature keys used in mass-balance checks
VM_KEY = "VolatileMatters/%"
FC_KEY = "FixedCarbon/%"
ASH_KEY = "Ash/%"
C_KEY = "C/%"
H_KEY = "H/%"
O_KEY = "O/%"
N_KEY = "N/%"
S_KEY = "S/%"

OXIDE_KEYS = [
    "Ash_SiO2",
    "Ash_Na2O",
    "Ash_MgO",
    "Ash_Al2O3",
    "Ash_K2O",
    "Ash_CaO",
    "Ash_P2O5",
    "Ash_CuO",
    "Ash_ZnO",
    "Ash_Fe2O3",
]


def _generate_chunk_worker(args: tuple) -> list[dict]:
    """Generate a chunk of Monte Carlo samples with a local independent RNG."""
    ranges, constants, chunk_size, child_seed = args
    rng = np.random.default_rng(child_seed)

    def get_range(feature: str):
        if feature in ranges.index:
            lo, hi = ranges.loc[feature, ["min", "max"]].values.tolist()
        else:
            val = constants.get(feature, np.nan)
            lo, hi = val, val
        return lo, hi

    def rand_in(lo, hi):
        if np.isnan(lo) or np.isnan(hi) or np.isclose(lo, hi):
            return lo if not np.isnan(lo) else hi
        return rng.uniform(lo, hi)

    def rand_int(lo, hi):
        if np.isnan(lo) or np.isnan(hi):
            return np.nan
        lo_i, hi_i = int(round(lo)), int(round(hi))
        if lo_i > hi_i:
            lo_i, hi_i = hi_i, lo_i
        return lo_i if lo_i == hi_i else rng.integers(lo_i, hi_i + 1)

    rows = []
    
    # Pre-cache ranges and properties to avoid slow repeated Pandas DataFrame indexing in the loop
    vm_range = get_range(VM_KEY)
    ash_range = get_range(ASH_KEY)
    fc_range = get_range(FC_KEY)
    c_range = get_range(C_KEY)
    h_range = get_range(H_KEY)
    n_range = get_range(N_KEY)
    s_range = get_range(S_KEY)
    o_range = get_range(O_KEY)

    oxide_ranges = {oxide: get_range(oxide) for oxide in OXIDE_KEYS}

    remaining_features = []
    for feature in ranges.index:
        canon_key = _canonical_no_space(feature)
        if canon_key.startswith("feedstocktype") or canon_key.startswith("mixingratio"):
            continue
        if feature in (VM_KEY, FC_KEY, ASH_KEY, C_KEY, H_KEY, O_KEY, N_KEY, S_KEY) or feature in OXIDE_KEYS:
            continue
        remaining_features.append((feature, canon_key, get_range(feature)))

    for _ in range(chunk_size):
        success = False
        for _ in range(5000):
            row: dict[str, float] = {}
            # Proximate analysis
            vm = rand_in(*vm_range)
            ash = rand_in(*ash_range)
            fc = 100.0 - vm - ash
            fc_lo, fc_hi = fc_range
            if fc < fc_lo or fc > fc_hi or fc < 0:
                continue
            row[VM_KEY] = vm
            row[FC_KEY] = fc
            row[ASH_KEY] = ash

            # Ultimate analysis
            c = rand_in(*c_range)
            h = rand_in(*h_range)
            n = rand_in(*n_range)
            s = rand_in(*s_range)
            o = 100.0 - (ash + c + h + n + s)
            o_lo, o_hi = o_range
            if o < o_lo or o > o_hi:
                continue
            row[C_KEY] = c
            row[H_KEY] = h
            row[N_KEY] = n
            row[S_KEY] = s
            row[O_KEY] = o

            # Oxides
            for oxide, ox_rng in oxide_ranges.items():
                val = rand_in(*ox_rng)
                if not np.isnan(val):
                    row[oxide] = val

            # Remaining features
            for feature, canon_key, feat_rng in remaining_features:
                if canon_key in INT_SAMPLED_FEATURES:
                    val = rand_int(*feat_rng)
                else:
                    val = rand_in(*feat_rng)
                if not np.isnan(val):
                    row[feature] = val

            # Fill missing with constants
            for feature, value in constants.items():
                if feature not in row or np.isnan(row.get(feature, np.nan)):
                    row[feature] = value

            rows.append(row)
            success = True
            break
        if not success:
            raise RuntimeError("Unable to generate a valid sample within max_attempts in worker process.")
    return rows


def build_monte_carlo_samples(
    ranges: pd.DataFrame, 
    constants: pd.Series, 
    n: int, 
    cores: int = 1, 
    seed: int = 2026
) -> pd.DataFrame:
    """Generate *n* Monte Carlo samples satisfying basic mass-balance constraints, in parallel if cores > 1."""
    if cores <= 1:
        print("Running sample generation sequentially (1 core) ...", flush=True)
        # Use single generator chunk worker directly
        rows = _generate_chunk_worker((ranges, constants, n, seed))
        return pd.DataFrame(rows)

    print(f"Spawning parallel generators across {cores} CPU cores ...", flush=True)
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(cores)

    # Divide work evenly
    chunk_size = n // cores
    extra = n % cores

    tasks = []
    for i in range(cores):
        current_chunk = chunk_size + (1 if i < extra else 0)
        tasks.append((ranges, constants, current_chunk, child_seeds[i]))

    with multiprocessing.Pool(processes=cores) as pool:
        chunk_results = pool.map(_generate_chunk_worker, tasks)

    # Flatten the list of lists
    all_rows = [row for chunk in chunk_results for row in chunk]
    return pd.DataFrame(all_rows)

# Non-sampled categories/constants
NON_SAMPLED_PREFIXES = ("feedstocktype", "mixingratio")
NON_SAMPLED_NAMES = {"location"}

# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monte Carlo prediction for Activation Energy (Ea) neural-network model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--samples", type=int, default=100_0000, help="Number of Monte-Carlo samples")
    parser.add_argument("--variation", type=float, default=0.10, help="Relative variation (e.g., 0.1 for ±10%) when only mean values are available")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for reproducibility")
    # Added --cores argument, defaulting to SLURM_CPUS_PER_TASK if on supercomputer
    default_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    parser.add_argument("--cores", type=int, default=default_cores, help="Number of CPU cores for parallel sample generation")
    args = parser.parse_args()

    n_samples: int = args.samples
    # We do NOT do legacy global np.random.seed(args.seed) for parallel generators, but still set it for main process jittering
    np.random.seed(args.seed)

    # Locate the trained .mat model
    mat_model_path = PROJECT_ROOT / "bpDNN4Ea_modelfiles" / "Results_trained.mat"
    if not mat_model_path.exists():
        raise FileNotFoundError(f"Expected {mat_model_path} not found.")

    # ------------------------------------------------------------------
    # Load model + training data (for scaling)
    # ------------------------------------------------------------------
    wrapper, feature_names, X_train, y_train = load_trained_model(mat_model_path)

    # ------------------------------------------------------------------
    # Locate Excel spreadsheets with parameter ranges and constants
    # ------------------------------------------------------------------
    range_path = PROJECT_ROOT / "data" / "raw" / "Municipal_Sludge_Data_cleaned_mean.xlsx"
    constant_path = PROJECT_ROOT / "data" / "raw" / "US_SewageSludge.xlsx"

    print("Reading parameter ranges …", flush=True)
    ranges_df = read_parameter_ranges(range_path)
    expand_single_point_ranges(ranges_df, rel_variation=args.variation)

    print("Reading constant feature values …", flush=True)
    constants_series = load_constant_features(constant_path)

    print(f"Generating {n_samples} Monte Carlo samples with {args.cores} core(s) …", flush=True)
    df_samples = build_monte_carlo_samples(ranges_df, constants_series, n_samples, cores=args.cores, seed=args.seed)

    # No column renaming required (Heating rate not present)

    # Overwrite any non-sampled categorical features with constants
    const_update = {
        feat: val
        for feat, val in constants_series.items()
        if str(feat).lower().startswith(NON_SAMPLED_PREFIXES)
        or _canonical_no_space(feat) in NON_SAMPLED_NAMES
    }
    if const_update:
        df_samples = df_samples.assign(**const_update)

    # Identify truly constant numeric columns – but NEVER jitter fixed identifiers
    # like Location or any FeedstockType_/MixingRatio_ features which must remain
    # exactly as specified in the constants spreadsheet.
    def _is_protected(col_name: str) -> bool:
        key = _canonical_no_space(col_name)
        return (
            key == "location"
            or key.startswith("feedstocktype")
            or key.startswith("mixingratio")
        )

    const_cols = [c for c in df_samples.columns if df_samples[c].std() == 0 and not _is_protected(c)]
    if const_cols:
        rng = np.random.default_rng(args.seed)
        jitter_data = {}
        for col in const_cols:
            base_val = df_samples[col].iloc[0]
            if np.isfinite(base_val) and base_val != 0:
                noise = rng.uniform(-0.05, 0.05, size=len(df_samples)) * abs(base_val)
                jitter_data[col] = np.clip(base_val + noise, 0, None)
            else:
                jitter_data[col] = rng.uniform(0, 0.05, size=len(df_samples))
        df_samples = df_samples.assign(**jitter_data)

    # ------------------------------------------------------------------
    # Align samples to model feature order and scale to [0,1] using training stats
    # ------------------------------------------------------------------
    X_raw = align_sample_features(df_samples, feature_names)

    train_min = X_train.min(axis=0)
    train_max = X_train.max(axis=0)
    denom = train_max - train_min
    denom[denom == 0] = 1.0

    X_scaled = np.clip((X_raw - train_min) / denom, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Predict Ea (scaled output) and inverse-transform back to kJ/mol.
    # ------------------------------------------------------------------
    y_pred_scaled = wrapper.predict(X_scaled)

    # Determine if outputs were scaled 0-1 or −1…1 in MATLAB training.
    y_train_min = y_train.min()
    y_train_max = y_train.max()
    y_range = y_train_max - y_train_min if y_train_max != y_train_min else 1.0

    if 0.0 <= y_train_min <= 1.0 and 0.0 <= y_train_max <= 1.0:
        # 0-1 scaling
        ea_real = y_pred_scaled * y_range + y_train_min
    else:
        # Assume −1…1 scaling
        ea_real = ((y_pred_scaled + 1.0) / 2.0) * y_range + y_train_min

    ea_series = pd.Series(ea_real, name="Ea_kJmol")

    # ------------------------------------------------------------------
    # Remove negative (non-physical) Ea values before saving/plotting
    # ------------------------------------------------------------------
    neg_mask = ea_series < 0
    if neg_mask.any():
        removed = int(neg_mask.sum())
        print(f"Removing {removed} sample(s) with negative Ea to retain physical plausibility …", flush=True)
        ea_series = ea_series.loc[~neg_mask]
        df_samples = df_samples.loc[~neg_mask]

    # ------------------------------------------------------------------
    # Persist results
    # ------------------------------------------------------------------
    output_dir = PROJECT_ROOT / "results" / "mc_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "mc_ea_predictions.csv"
    df_out = df_samples.copy()
    df_out["Ea_kJmol"] = ea_series
    df_out.to_csv(csv_path, index=False)
    print(f"Saved predictions CSV → {csv_path}", flush=True)

    # Plot distribution
    plot_path = output_dir / "mc_ea_distribution.png"
    plot_ea_distribution(ea_series, plot_path)
    print(f"Saved distribution plot → {plot_path} (+ .svg)", flush=True)


if __name__ == "__main__":
    main() 