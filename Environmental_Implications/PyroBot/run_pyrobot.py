"""
PyroBot: Unified CLI Entry-Point & Conversational Agent Shell
============================================================
This script is the main executable for the PyroBot framework. It provides:
1. --mode mc: High-performance parallelized Monte Carlo predictions (both Ea & Yields).
2. --mode optimize: Simplex Scipy dual-scenario (A & B) continuous optimization sweeps.
3. --mode agent: Interactive scientific Chatbot shell powered by local Qwen3.6 Brain.
"""

from __future__ import annotations

# Set thread isolation environment variables before importing heavy packages
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress duplicate variable warnings from SciPy
warnings.filterwarnings("ignore", message="Duplicate variable name")

from core.dnn_surrogates import (
    load_trained_model,
    scale_features,
    unscale_outputs,
    canonicalize_feature_name,
    MatlabNeuralNetworkWrapper,
)
from core.continuous_optimizer import (
    optimize_multi_feedstock_blend,
    optimize_single_additive_blend,
    load_feedstock_names,
)
from agent.pyrobot_orchestrator import PyroBotOrchestrator

# -----------------------------------------------------------------------------
# 1. Parallelized Monte Carlo Scanner Implementation
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
    "Ash_SiO2", "Ash_Na2O", "Ash_MgO", "Ash_Al2O3", "Ash_K2O",
    "Ash_CaO", "Ash_P2O5", "Ash_CuO", "Ash_ZnO", "Ash_Fe2O3"
]

def read_parameter_ranges(range_path: Path) -> pd.DataFrame:
    """Read parameter boundaries from range spreadsheet."""
    df = pd.read_excel(range_path, header=0)
    feat_col = df.columns[0]
    
    # Try to find min and max columns
    min_col = [c for c in df.columns if any(k in str(c).lower() for k in ["min", "lower", "low"])][0]
    max_col = [c for c in df.columns if any(k in str(c).lower() for k in ["max", "upper", "high"])][0]
    
    ranges = df[[feat_col, min_col, max_col]].copy()
    ranges.columns = ["feature", "min", "max"]
    ranges["min"] = pd.to_numeric(ranges["min"], errors="coerce")
    ranges["max"] = pd.to_numeric(ranges["max"], errors="coerce")
    ranges.set_index("feature", inplace=True)
    
    ranges.index = [canonicalize_feature_name(idx) for idx in ranges.index]
    return ranges[~ranges.index.duplicated(keep="first")]

def load_constant_features(constant_path: Path) -> pd.Series:
    """Read static sewage sludge properties."""
    df = pd.read_excel(constant_path, header=0)
    const_series = df.iloc[0] if len(df) == 1 else df.mean()
    const_series.index = [canonicalize_feature_name(idx) for idx in const_series.index]
    return const_series[~const_series.index.duplicated(keep="first")]

def _mc_chunk_worker(args: tuple) -> list[dict]:
    """Independent process worker generating constrained proximate/ultimate samples."""
    ranges, constants, chunk_size, child_seed = args
    rng = np.random.default_rng(child_seed)
    
    def random_val(feat: str):
        if feat in ranges.index:
            lo, hi = ranges.loc[feat, ["min", "max"]].values
            if np.isnan(lo) or np.isnan(hi) or np.isclose(lo, hi):
                return lo if not np.isnan(lo) else hi
            return rng.uniform(lo, hi)
        return constants.get(feat, np.nan)

    rows = []
    for _ in range(chunk_size):
        for _ in range(5000): # max attempts
            row = {}
            # Industrial Proximate balance: VM + FC + Ash = 100
            vm = random_val(VM_KEY)
            ash = random_val(ASH_KEY)
            fc = 100.0 - vm - ash
            if fc < 0:
                continue
                
            row.update({VM_KEY: vm, FC_KEY: fc, ASH_KEY: ash})
            
            # Stoichiometric Ultimate balance: Ash + C + H + O + N + S = 100
            c = random_val(C_KEY)
            h = random_val(H_KEY)
            n = random_val(N_KEY)
            s = random_val(S_KEY)
            o = 100.0 - (ash + c + h + n + s)
            if o < 0:
                continue
                
            row.update({C_KEY: c, H_KEY: h, N_KEY: n, S_KEY: s, O_KEY: o})
            
            # Oxide mineral ratios
            for oxide in OXIDE_KEYS:
                row[oxide] = random_val(oxide)
                
            # Populate remaining features
            for feature in ranges.index:
                if feature not in row and not str(feature).lower().startswith(("feedstock", "mixing")):
                    row[feature] = random_val(feature)
                    
            # Inject general locked defaults
            for feature, value in constants.items():
                if feature not in row or np.isnan(row[feature]):
                    row[feature] = value
                    
            rows.append(row)
            break
            
    return rows

def generate_mc_samples(ranges: pd.DataFrame, constants: pd.Series, n: int, cores: int, seed: int) -> pd.DataFrame:
    """Spawns parallel generator chunks across cores."""
    if cores <= 1:
        return pd.DataFrame(_mc_chunk_worker((ranges, constants, n, seed)))
        
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(cores)
    chunk = n // cores
    extra = n % cores
    
    tasks = []
    for i in range(cores):
        curr_chunk = chunk + (1 if i < extra else 0)
        tasks.append((ranges, constants, curr_chunk, child_seeds[i]))
        
    with multiprocessing.Pool(processes=cores) as pool:
        results = pool.map(_mc_chunk_worker, tasks)
        
    return pd.DataFrame([row for chunk_res in results for row in chunk_res])

def run_mc_predictions_pipeline(n_samples: int, seed: int, cores: int, project_root: Path, out_dir_arg: str = ""):
    """Run parallelized Monte Carlo prediction simulation for both Ea and Yields."""
    print(f"\n[Monte Carlo Pipeline] Scanning baseline sewage sludge properties ({n_samples} samples)...")
    
    # Path anchoring
    range_path = project_root / "data" / "raw" / "Municipal_Sludge_Data_cleaned_mean.xlsx"
    constant_path = project_root / "data" / "raw" / "US_SewageSludge.xlsx"
    ea_mat = project_root / "models" / "bpDNN2Ea" / "Results_trained.mat"
    yield_mat = project_root / "models" / "bpDNN2Yield" / "Results_trained.mat"
    
    ranges = read_parameter_ranges(range_path)
    constants = load_constant_features(constant_path)
    
    # Exclude co-pyrolysis locks from random sweep
    to_drop = [c for c in ranges.index if str(c).lower().startswith(("feedstocktype", "mixingratio", "location"))]
    ranges.drop(index=to_drop, errors="ignore", inplace=True)
    
    print(f"Generating samples on {cores} CPU cores...", flush=True)
    df_samples = generate_mc_samples(ranges, constants, n_samples, cores, seed)
    
    # Calculate ReactionTime dynamically
    df_samples["ReactionTime/min"] = df_samples["TargetTemperature/Celsius"] / df_samples["Heating rate"]
    df_samples = df_samples.fillna(0.0)
    
    # RENAME to match NN headers
    RENAME_DICT = {"Heating rate": "HeatingRate/(K/min)"}
    df_samples.rename(columns=RENAME_DICT, inplace=True)
    
    # 1. Predict apparent Ea
    print("Loading pre-trained apparent Ea network...", flush=True)
    ea_wrapper, ea_features, ea_train_X, ea_train_y = load_trained_model(ea_mat)
    
    # Align and Scale features
    ea_raw = np.zeros((len(df_samples), len(ea_features)))
    for i, col in enumerate(ea_features):
        if col in df_samples.columns:
            ea_raw[:, i] = df_samples[col].values
            
    ea_scaled = scale_features(ea_raw, ea_train_X)
    ea_pred_norm = ea_wrapper.predict(ea_scaled)
    ea_real = unscale_outputs(ea_pred_norm, ea_train_y)
    
    # 2. Predict Product Yields
    print("Loading pre-trained product yield network...", flush=True)
    yield_wrapper, yield_features, yield_train_X, yield_train_y = load_trained_model(yield_mat)
    
    yield_raw = np.zeros((len(df_samples), len(yield_features)))
    for i, col in enumerate(yield_features):
        if col in df_samples.columns:
            yield_raw[:, i] = df_samples[col].values
            
    yield_scaled = scale_features(yield_raw, yield_train_X)
    
    yield_preds = []
    for out_idx in range(3):
        yield_wrapper.target_idx = out_idx
        norm_pred = yield_wrapper.predict(yield_scaled)
        yield_preds.append(unscale_outputs(norm_pred, yield_train_y, out_idx))
        
    # Compile final predictions df
    pred_df = pd.DataFrame({
        "Ea": ea_real,
        "Biochar": yield_preds[0],
        "Bioliquid": yield_preds[1],
        "Biogas": yield_preds[2]
    })
    
    # Enforce physical bounds (remove unphysical negative samples)
    valid_mask = (pred_df >= 0).all(axis=1)
    df_samples = df_samples[valid_mask].reset_index(drop=True)
    pred_df = pred_df[valid_mask].reset_index(drop=True)
    
    # Export CSV results
    if out_dir_arg:
        out_dir = Path(out_dir_arg).resolve()
    else:
        out_dir = project_root / "results" / "mc_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "mc_predictions_complete.csv"
    
    pd.concat([df_samples, pred_df], axis=1).to_csv(csv_path, index=False)
    print(f"Saved complete prediction spreadsheet → {csv_path}")
    
    # Generate violin distribution plot
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Ea distribution
    sns.violinplot(data=pred_df, y="Ea", color="skyblue", inner="quartile", ax=axes[0])
    axes[0].set_title("Apparent Activation Energy (Ea)")
    axes[0].set_ylabel("kJ/mol")
    
    # Yields distribution
    yield_melt = pred_df[["Biochar", "Bioliquid", "Biogas"]].melt(var_name="Product", value_name="Yield (%)")
    sns.violinplot(data=yield_melt, x="Product", y="Yield (%)", palette="Set2", inner="quartile", ax=axes[1])
    axes[1].set_title("Product Yield Distribution")
    
    plt.suptitle("PyroBot Monte Carlo Baseline Scan Results", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    fig_path = out_dir / "mc_uncertainty_distributions.png"
    fig.savefig(fig_path, dpi=300)
    fig.savefig(fig_path.with_suffix(".svg"), format="svg")
    plt.close(fig)
    print(f"Saved publication-quality uncertainty plot → {fig_path} (+ .svg)")


# -----------------------------------------------------------------------------
# 2. Continuous Sweeper Optimizer Pipeline
# -----------------------------------------------------------------------------
def run_blending_optimizations_pipeline(scenario: str, cores: int, project_root: Path, out_dir_arg: str = ""):
    """Run Scipy SLSQP continuous optimization sweep over Scenario A or B parameters."""
    print(f"\n[Optimizer Pipeline] Running recipe sweep under Scenario {scenario}...")
    
    # Align scenario caps
    if scenario.upper() == "A":
        sludge_ratio = 0.50
        max_ratio = 0.25
        out_folder = "blending_outputs_50"
    else:
        sludge_ratio = 0.80
        max_ratio = 0.10
        out_folder = "blending_outputs_20"
        
    ref_xlsx = project_root / "data" / "reference" / "Feedstock_types_compiled.xlsx"
    f_names = load_feedstock_names(ref_xlsx)
    
    # Selected highly synergistic candidates list
    # Aligned with physical properties: Almond shell, Walnut shell, Rice husk, Pine wood, etc.
    active_ids = [11, 14, 28, 32, 41, 45, 65, 70, 74]
    
    # Loading models
    ea_mat = project_root / "models" / "bpDNN2Ea" / "Results_trained.mat"
    yield_mat = project_root / "models" / "bpDNN2Yield" / "Results_trained.mat"
    
    ea_wrapper, ea_features, ea_train_X, ea_train_y = load_trained_model(ea_mat)
    yield_wrapper, yield_features, yield_train_X, yield_train_y = load_trained_model(yield_mat)
    
    # Construct base sewage sludge properties vector
    constant_path = project_root / "data" / "raw" / "US_SewageSludge.xlsx"
    constants = load_constant_features(constant_path)
    
    base_vec = np.zeros(len(ea_features))
    for i, col in enumerate(ea_features):
        if col in constants.index:
            base_vec[i] = constants[col]
            
    # Calculate optimal blends sequentially or in parallel
    print(f"Running continuous SLSQP optimizations across candidate promoters...", flush=True)
    
    # We will search the optimal ternary combinations (2 additives + sewage sludge)
    import itertools
    combos = list(itertools.combinations(active_ids, 2))
    
    rows = []
    for combo in combos:
        r, stats = optimize_multi_feedstock_blend(
            active_additive_ids=list(combo),
            sludge_ratio=sludge_ratio,
            max_individual_ratio=max_ratio,
            baseline_vector=base_vec,
            feature_names=ea_features,
            ea_wrapper=ea_wrapper,
            ea_train_X=ea_train_X,
            ea_train_y=ea_train_y,
            yield_wrapper=yield_wrapper,
            yield_train_X=yield_train_X,
            yield_train_y=yield_train_y,
            objective="pareto_joint"
        )
        
        row = {
            "Combo_IDs": "-".join(map(str, combo)),
            "Combo_Names": " + ".join([f_names.get(cid, "Unknown") for cid in combo]),
            "Additive_1_Ratio": r[0],
            "Additive_2_Ratio": r[1],
            "Sludge_Ratio": sludge_ratio,
            "Apparent_Ea": stats["Ea"],
            "Predicted_Char_Yield": stats["Char_Yield"],
            "Predicted_Liquid_Yield": stats["Liquid_Yield"]
        }
        rows.append(row)
        
    df_results = pd.DataFrame(rows).sort_values("Apparent_Ea", ascending=True)
    
    if out_dir_arg:
        out_dir = Path(out_dir_arg).resolve()
    else:
        out_dir = project_root / "results" / "optimized_blends" / out_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "combo_optimized_recipes.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"Saved optimized co-pyrolysis recipe sheet → {csv_path}")
    
    # Generate optimal recipe bar comparison plot
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    top_10 = df_results.head(10)
    sns.barplot(data=top_10, x="Apparent_Ea", y="Combo_Names", palette="viridis", ax=ax)
    ax.set_title(f"Top 10 Catalytic Co-pyrolysis Recipes (Scenario {scenario})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Apparent Ea (kJ/mol)")
    ax.set_ylabel("Recipe Blend Candidates")
    plt.tight_layout()
    
    fig_path = out_dir / "recipe_optimizations_comparison.png"
    fig.savefig(fig_path, dpi=300)
    fig.savefig(fig_path.with_suffix(".svg"), format="svg")
    plt.close(fig)
    print(f"Saved publication-quality optimizations plot → {fig_path} (+ .svg)")


# -----------------------------------------------------------------------------
# 3. Main Entry Shell
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="PyroBot: Central Computational orchestrator CLI tool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", type=str, required=True, choices=["mc", "optimize", "agent"], help="Active workflow execution mode")
    parser.add_argument("--samples", type=int, default=2000000, help="Number of Monte Carlo samples")
    parser.add_argument("--seed", type=int, default=2026, help="RNG seed")
    parser.add_argument("--scenario", type=str, default="B", choices=["A", "B"], help="Optimization locked scenario limits")
    parser.add_argument("--cores", type=int, default=1, help="CPU cores allocated")
    parser.add_argument("--chatbot", action="store_true", help="Launch interactive Chatbot Shell in agent mode")
    parser.add_argument("--out-dir", type=str, default="", help="Custom output directory for results")
    parser.add_argument(
        "--query",
        type=str,
        default="Design a ternary co-pyrolysis recipe with municipal sewage sludge that maximizes Biochar yield above 42% at a low target temperature of 450°C, while keeping the Apparent Activation Energy below 390 kJ/mol under Scenario B (80% sludge load) constraints.",
        help="Command-line query for inverse design when chatbot mode is disabled"
    )
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent
    
    if args.mode == "mc":
        run_mc_predictions_pipeline(args.samples, args.seed, args.cores, project_root, args.out_dir)
        
    elif args.mode == "optimize":
        run_blending_optimizations_pipeline(args.scenario, args.cores, project_root, args.out_dir)
        
    elif args.mode == "agent":
        orchestrator = PyroBotOrchestrator(cores=args.cores, results_dir=Path(args.out_dir) if args.out_dir else None)
        
        if args.chatbot:
            print("\n" + "="*80)
            print("PyroBot Qwen3.6-35B-A3B Autonomous Chatbot Shell Initialized.")
            print("Type 'exit' or 'quit' to terminate the session.")
            print("="*80)
            
            while True:
                try:
                    user_input = input("\nResearcher 👨‍🔬: ")
                    if user_input.lower() in ["exit", "quit"]:
                        print("Terminating conversational session. Happy computing!")
                        break
                    if not user_input.strip():
                        continue
                        
                    print("\nPyroBot 🤖 is thinking...")
                    # Automatically call full closed-loop inverse design if query implies blending design
                    if any(k in user_input.lower() for k in ["design", "optimize", "blend", "recipe", "ratio"]):
                        report = orchestrator.execute_inverse_design(user_input)
                        print(report)
                    else:
                        ans = orchestrator.run_agent_query(
                            system_prompt="You are a brilliant pyrolysis assistant. Be concise.",
                            user_prompt=user_input
                        )
                        print(f"\nPyroBot 🤖: {ans}")
                except (KeyboardInterrupt, EOFError):
                    break
        else:
            # Command mode inverse design sweep using provided or default target query
            user_input = args.query
            print(f"\n[Command Mode] Submitted target objective query to PyroBot Agent:\n\"{user_input}\"\n")
            report = orchestrator.execute_inverse_design(user_input)
            print(report)


if __name__ == "__main__":
    main()
