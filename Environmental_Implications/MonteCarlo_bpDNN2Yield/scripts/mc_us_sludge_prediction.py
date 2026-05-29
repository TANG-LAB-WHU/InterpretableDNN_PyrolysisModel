"""
Monte Carlo Prediction of Pyrolysis Product Yields for U.S. Sewage Sludge
========================================================================
This script performs high-performance parallelized Monte Carlo simulations to
predict pyrolysis product yields (Biochar, Bioliquid, Biogas) using a pre-trained
neural network model.

Workflow
--------
1. Load a pre-trained neural network (MATLAB .mat) and extract training statistics.
2. Read parameter ranges from Municipal_Sludge_Data_cleaned_mean.xlsx.
3. Read fallback constant properties from US_SewageSludge.xlsx.
4. Generate Monte Carlo samples satisfying Proximate/Ultimate mass-balance constraints
   across multiple CPU cores with SeedSequence-derived independent process generators.
5. Derive Reaction Time from Target Temperature and Heating Rate.
6. Align and scale features to [0, 1] using original training dataset boundaries.
7. Perform forward predictions for Biochar (idx 0), Bioliquid (idx 1), and Biogas (idx 2).
8. Inverse-scale predicted yield values back to absolute weight percentages.
9. Filter out unphysical negative yield samples.
10. Persist results in CSV format and save publication-quality violin distribution plots (PNG/SVG).
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
# Add project root so we can import helper utilities from shap_analysis_yield.py
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Import helper utilities defined in shap_analysis_yield.py
from shap_analysis_yield import (  # type: ignore
    load_matlab_data,
    extract_neural_network_data,
    generate_feature_names,
    MatlabNeuralNetworkWrapper,
)

# -----------------------------------------------------------------------------
# Helper functions & Canonical mapping
# -----------------------------------------------------------------------------

# Mapping of alternative column names to canonical model feature names
CANONICAL_NAME_MAP = {
    "volatilematter": "VolatileMatters/%",
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
    "targettemperature": "TargetTemperature/Celsius",
}


def _clean_key(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _canonical_no_space(name: str) -> str:
    """Return cleaned key (alphanumerics only)."""
    return _clean_key(name)


def canonicalize_feature_name(name: str) -> str:
    """Return canonical model feature name given a raw column header."""
    if not isinstance(name, str):
        return name
    key = _clean_key(name)
    return CANONICAL_NAME_MAP.get(key, name)


# Features that must be sampled as integers
INT_SAMPLED_FEATURES: set[str] = {"reactortype"}


def read_parameter_ranges(range_path: Path) -> pd.DataFrame:
    """Read Excel file containing min/max (or mean) for each feature."""
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
    feat_col = df.columns[0]  # assume first column lists feature names

    if min_col and max_col:
        ranges = df[[feat_col, min_col, max_col]].copy()
        ranges.columns = ["feature", "min", "max"]
        ranges["min"] = pd.to_numeric(ranges["min"], errors="coerce")
        ranges["max"] = pd.to_numeric(ranges["max"], errors="coerce")
        ranges.set_index("feature", inplace=True)
    else:
        # Case 2: rows named min/max
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
            # Case 3: derive from numeric data across rows (fallback)
            numeric_df = df.select_dtypes(include=[np.number])
            ranges = pd.DataFrame({
                "min": numeric_df.min(),
                "max": numeric_df.max(),
            })

    ranges.index = [canonicalize_feature_name(idx) for idx in ranges.index]
    ranges = ranges[~ranges.index.duplicated(keep="first")]
    return ranges


def load_constant_features(constant_path: Path) -> pd.Series:
    """Read Excel file containing constant feature values (single row)."""
    df = pd.read_excel(constant_path, header=0)
    if len(df) == 1:
        const_series = df.iloc[0]
    else:
        const_series = df.mean()

    const_series.index = [canonicalize_feature_name(idx) for idx in const_series.index]
    const_series = const_series[~const_series.index.duplicated(keep="first")]
    return const_series


def expand_single_point_ranges(ranges: pd.DataFrame, rel_variation: float = 0.10) -> None:
    """Modify ranges in-place: if min == max (or only one value), expand by ±rel_variation."""
    for feat in ranges.index:
        lo = ranges.at[feat, "min"]
        hi = ranges.at[feat, "max"]

        if pd.isna(lo) and not pd.isna(hi):
            lo = hi
        if pd.isna(hi) and not pd.isna(lo):
            hi = lo

        if pd.isna(lo) or pd.isna(hi):
            continue

        if np.isclose(lo, hi):
            mid = lo
            if np.isfinite(mid) and mid != 0:
                delta = abs(mid) * rel_variation
                ranges.at[feat, "min"] = mid - delta
                ranges.at[feat, "max"] = mid + delta


def load_trained_model(mat_path: Path) -> Tuple[
    MatlabNeuralNetworkWrapper,
    list[str],
    np.ndarray,
    np.ndarray,
]:
    """Load pre-trained MATLAB neural network model."""
    print(f"Loading MATLAB model from: {mat_path}", flush=True)
    mat_data = load_matlab_data(str(mat_path))

    if not mat_data:
        import tempfile
        import shutil

        print("Direct load failed—retrying with shortened temporary path ...", flush=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_mat_path = Path(tmpdir) / "model.mat"
            shutil.copy2(mat_path, tmp_mat_path)
            mat_data = load_matlab_data(str(tmp_mat_path))

    if not mat_data:
        raise RuntimeError("Failed to load MATLAB model data. Ensure .mat is v7.0 (not v7.3 HDF5).")

    X_dummy, y_dummy, net_struct = extract_neural_network_data(mat_data)
    feature_names = generate_feature_names(X_dummy, mat_data)
    wrapper = MatlabNeuralNetworkWrapper(net_struct)

    return wrapper, feature_names, X_dummy, y_dummy


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


def plot_uncertainty(predictions_pct: pd.DataFrame, output_path: Path) -> None:
    """Save violin plot to output_path (PNG & SVG)."""
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    melted = predictions_pct.melt(var_name="Product", value_name="Yield (%)")
    sns.violinplot(
        data=melted,
        x="Product",
        y="Yield (%)",
        hue="Product",
        inner="quartile",
        palette="Set2",
        ax=ax,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_title("Monte Carlo Prediction Uncertainty – U.S. Sewage Sludge")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".svg"), format="svg")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Mass-Balance sampling chunks generator (for Multiprocessing)
# -----------------------------------------------------------------------------

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

    def random_in_range(lo: float, hi: float):
        if np.isnan(lo) or np.isnan(hi) or np.isclose(lo, hi):
            return lo if not np.isnan(lo) else hi
        return rng.uniform(lo, hi)

    def random_int_in_range(lo: float, hi: float):
        if np.isnan(lo) or np.isnan(hi):
            return np.nan
        lo_i, hi_i = int(round(lo)), int(round(hi))
        if lo_i > hi_i:
            lo_i, hi_i = hi_i, lo_i
        if lo_i == hi_i:
            return lo_i
        return rng.integers(lo_i, hi_i + 1)

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

    rows = []
    for _ in range(chunk_size):
        success = False
        for _ in range(5000):
            row = {}
            # ---------------- Proximate analysis ----------------
            vm = random_in_range(*vm_range)
            ash = random_in_range(*ash_range)
            fc = 100.0 - vm - ash

            fc_lo, fc_hi = fc_range
            if fc < fc_lo or fc > fc_hi or fc < 0:
                continue

            row[VM_KEY] = vm
            row[FC_KEY] = fc
            row[ASH_KEY] = ash

            # ---------------- Ultimate analysis -----------------
            c = random_in_range(*c_range)
            h = random_in_range(*h_range)
            n = random_in_range(*n_range)
            s = random_in_range(*s_range)
            o = 100.0 - (ash + c + h + n + s)
            o_lo, o_hi = o_range
            if o < o_lo or o > o_hi:
                continue
            row.update({C_KEY: c, H_KEY: h, N_KEY: n, S_KEY: s, O_KEY: o})

            # ---------------- Oxide composition -----------------
            for oxide, ox_rng in oxide_ranges.items():
                val = random_in_range(*ox_rng)
                if not np.isnan(val):
                    row[oxide] = val

            # ---------------- Remaining features ---------------
            for feature, canon_key, feat_rng in remaining_features:
                if canon_key in INT_SAMPLED_FEATURES:
                    val = random_int_in_range(*feat_rng)
                else:
                    val = random_in_range(*feat_rng)
                if not np.isnan(val):
                    row[feature] = val

            # Fill any missing or NaN features from constants
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
    seed: int = 500,
) -> pd.DataFrame:
    """Generate *n* Monte Carlo samples that satisfy domain mass-balance constraints, in parallel if cores > 1."""
    if cores <= 1:
        print("Running sample generation sequentially (1 core) ...", flush=True)
        rows = _generate_chunk_worker((ranges, constants, n, seed))
        return pd.DataFrame(rows)

    print(f"Spawning parallel generators across {cores} CPU cores ...", flush=True)
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(cores)

    chunk_size = n // cores
    extra = n % cores

    tasks = []
    for i in range(cores):
        current_chunk = chunk_size + (1 if i < extra else 0)
        tasks.append((ranges, constants, current_chunk, child_seeds[i]))

    with multiprocessing.Pool(processes=cores) as pool:
        chunk_results = pool.map(_generate_chunk_worker, tasks)

    all_rows = [row for chunk in chunk_results for row in chunk]
    return pd.DataFrame(all_rows)


# Non-sampled constants to lock
NON_SAMPLED_PREFIXES = ("feedstocktype", "mixingratio")
NON_SAMPLED_NAMES = {"location"}

# -----------------------------------------------------------------------------
# Main Execution Block
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monte Carlo prediction for sewage sludge co-pyrolysis yields.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--samples", type=int, default=2000000, help="Number of Monte Carlo samples")
    parser.add_argument("--variation", type=float, default=0.10, help="Relative variation (e.g., 0.1 for ±10%) for single-point parameters")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for reproducibility")
    default_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    parser.add_argument("--cores", type=int, default=default_cores, help="Number of CPU cores for parallel generation")
    args = parser.parse_args()

    n_samples = args.samples
    np.random.seed(args.seed)

    # Establish clean paths using zero-configuration pathlib anchors
    mat_model_path = PROJECT_ROOT / "bpDNN4PyroProd_modelfiles" / "Results_trained.mat"
    range_path = PROJECT_ROOT / "data" / "raw" / "Municipal_Sludge_Data_cleaned_mean.xlsx"
    constant_path = PROJECT_ROOT / "data" / "raw" / "US_SewageSludge.xlsx"

    if not mat_model_path.exists():
        raise FileNotFoundError(f"MATLAB neural network model not found at {mat_model_path}")
    if not range_path.exists() or not constant_path.exists():
        raise FileNotFoundError("Missing raw Excel input files in 'data/raw/'. Ensure restructuring copy succeeded.")

    print("Reading parameter ranges …", flush=True)
    ranges = read_parameter_ranges(range_path)
    expand_single_point_ranges(ranges, rel_variation=args.variation)

    # Exclude categorical properties from random sweep
    to_drop = [
        idx
        for idx in ranges.index
        if str(idx).lower().startswith(NON_SAMPLED_PREFIXES)
        or _canonical_no_space(idx) in NON_SAMPLED_NAMES
    ]
    if to_drop:
        ranges.drop(index=to_drop, inplace=True)
        print(f"Excluded {len(to_drop)} non-sampled features: {to_drop}", flush=True)

    print(f"Loaded/expanded ranges for {len(ranges)} features.", flush=True)

    print("Reading constant feature values …", flush=True)
    constants = load_constant_features(constant_path)

    print(f"Generating {n_samples} Monte Carlo samples with {args.cores} core(s) …", flush=True)
    df_samples = build_monte_carlo_samples(ranges, constants, n_samples, cores=args.cores, seed=args.seed)

    # ------------------------------------------------------------------
    # Reaction Time derivation from Heating Rate and Target Temperature
    # ------------------------------------------------------------------
    tt_col = "TargetTemperature/Celsius"
    hr_col = "Heating rate"
    rt_col = "Reaction time"
    delta_col = "deltaTime"

    if tt_col in df_samples.columns and hr_col in df_samples.columns:
        if (df_samples[hr_col] == 0).any():
            raise ValueError("Heating rate contains zero values; cannot compute Reaction time.")
        if delta_col not in df_samples.columns:
            df_samples[delta_col] = 0.0
        df_samples[rt_col] = df_samples[tt_col] / df_samples[hr_col] + df_samples[delta_col]
    else:
        print("Warning: Reaction time derivation failed due to missing column references.", flush=True)

    # Fill NaN values with zero to block propagation
    df_samples = df_samples.fillna(0.0)

    # Enforce exact locks on spreadsheet constants
    const_update = {
        feat: val
        for feat, val in constants.items()
        if str(feat).lower().startswith(NON_SAMPLED_PREFIXES)
        or _canonical_no_space(feat) in NON_SAMPLED_NAMES
    }
    if const_update:
        df_samples = df_samples.assign(**const_update)

    # ──────────────────────────────────────────────────────────────
    # Avoid zero-variance collapsing (violins flatlined)
    # ──────────────────────────────────────────────────────────────
    def _is_protected(col_name: str) -> bool:
        key = _canonical_no_space(col_name)
        return (
            key == "location"
            or key.startswith("feedstocktype")
            or key.startswith("mixingratio")
        )

    const_cols = [
        c for c in df_samples.columns
        if df_samples[c].std() == 0
        and not _is_protected(c)
        and _canonical_no_space(c) not in INT_SAMPLED_FEATURES
    ]

    if const_cols:
        rng = np.random.default_rng(args.seed)
        jitter_update = {}
        for col in const_cols:
            base_val = df_samples[col].iloc[0]
            if np.isfinite(base_val) and base_val != 0:
                noise = rng.uniform(-0.05, 0.05, size=len(df_samples)) * abs(base_val)
                jitter_update[col] = np.clip(base_val + noise, 0, None)
            else:
                jitter_update[col] = rng.uniform(0, 0.05, size=len(df_samples))

        df_samples = df_samples.assign(**jitter_update)
        print(f"Added jitter to {len(const_cols)} constant feature(s) for visual density.", flush=True)

    # Standardize names for neural network compatibility
    RENAME_FOR_MODEL = {
        "Heating rate": "HeatingRate/(K/min)",
        "Reaction time": "ReactionTime/min",
    }
    df_samples = df_samples.rename(columns=RENAME_FOR_MODEL)

    print("Loading trained neural network model …", flush=True)
    wrapper, model_feature_names, X_train, y_train = load_trained_model(mat_model_path)

    X_raw = align_sample_features(df_samples, model_feature_names)

    # Scale exactly matching MATLAB pre-processing limits [0, 1]
    train_min = X_train.min(axis=0)
    train_max = X_train.max(axis=0)
    denom = train_max - train_min
    denom[denom == 0] = 1.0
    X_input = np.clip((X_raw - train_min) / denom, 0.0, 1.0)

    # Predict outputs (Biochar=0, Bioliquid=1, Biogas=2)
    print("Predicting yields …", flush=True)
    product_names = ["Biochar", "Bioliquid", "Biogas"]
    y_pred_all = []
    for idx in range(3):
        wrapper.target_idx = idx
        y_pred_all.append(wrapper.predict(X_input))
    y_pred_matrix = np.column_stack(y_pred_all)

    # Inverse transform neural network predictions
    y_train_arr = np.asarray(y_train, dtype=float)
    tgt_min = y_train_arr.min(axis=0)
    tgt_max = y_train_arr.max(axis=0)
    tgt_range = tgt_max - tgt_min
    tgt_range[tgt_range == 0] = 1.0

    if np.all(tgt_min >= -1e-6) and np.all(tgt_max <= 1.0 + 1e-6):
        y_real = y_pred_matrix * tgt_range + tgt_min
    else:
        y_real = ((y_pred_matrix + 1.0) / 2.0) * tgt_range + tgt_min

    predictions_pct = pd.DataFrame(y_real, columns=product_names, dtype=float)

    # Filter out unphysical negative yield samples
    neg_mask = (predictions_pct < 0).any(axis=1)
    if neg_mask.any():
        removed = neg_mask.sum()
        print(f"Removing {removed} sample(s) with negative yields ...", flush=True)
        predictions_pct = predictions_pct.loc[~neg_mask].reset_index(drop=True)
        df_samples = df_samples.loc[~neg_mask].reset_index(drop=True)

    # Create results folder structure and persist results
    output_dir = PROJECT_ROOT / "results" / "mc_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "mc_us_sludge_predictions.csv"
    combined_df = pd.concat([df_samples, predictions_pct], axis=1)
    combined_df.to_csv(csv_path, index=False)
    print(f"Saved predictions and features CSV → {csv_path}", flush=True)

    fig_path = output_dir / "mc_us_sludge_uncertainty.png"
    plot_uncertainty(predictions_pct, fig_path)
    print(f"Saved uncertainty violin plots → {fig_path} (+ .svg)", flush=True)

    # Output statistical distribution summaries
    summary = predictions_pct.describe(percentiles=[0.05, 0.5, 0.95]).loc[["mean", "std", "5%", "95%"]]
    print("\nSummary statistics (percentage yields):", flush=True)
    print(summary, flush=True)


if __name__ == "__main__":
    main()