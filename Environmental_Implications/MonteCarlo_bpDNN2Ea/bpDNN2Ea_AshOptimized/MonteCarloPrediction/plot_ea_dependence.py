#!/usr/bin/env python3
"""Plot Ea dependence scatter plot from Monte-Carlo predictions.

This script reads *mc_ea_predictions.csv* and produces a dependence plot
that mimics a SHAP dependence plot:
  • x-axis: selected feature value (default: Ash_SiO2)
  • y-axis: Ea predictions (kJ/mol)
  • point colour: second feature value (default: Ash_Na2O)

The output figure is saved as both PNG and SVG in the same folder as the CSV.
Only English is used throughout this file as requested.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# -----------------------------------------------------------------------------
# Data export helper
# -----------------------------------------------------------------------------


def export_plot_data(
    df: pd.DataFrame,
    x_feature: str,
    y_feature: str,
    color_feature: str,
    csv_path: Path,
) -> None:
    """Save selected columns to *csv_path*."""
    cols = [x_feature, y_feature, color_feature]
    df[cols].to_csv(csv_path, index=False)
    print(f"Saved underlying data → {csv_path}")


# -----------------------------------------------------------------------------
# Helper function
# -----------------------------------------------------------------------------

def draw_dependence(
    df: pd.DataFrame,
    x_feature: str,
    y_feature: str,
    color_feature: str,
    out_path: Path,
) -> None:
    """Create and save a dependence scatter plot.

    Args:
        df: DataFrame containing at least the specified columns.
        x_feature: Column name for the x-axis.
        y_feature: Column name for the y-axis.
        color_feature: Column name used for point colours.
        out_path: Base path (without extension) for saving the figure.
    """
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    cmap = sns.color_palette("coolwarm", as_cmap=True)
    scatter = ax.scatter(
        df[x_feature],
        df[y_feature],
        c=df[color_feature],
        cmap=cmap,
        s=22,  # point size similar to SHAP plot
        linewidths=0,
        alpha=0.8,
    )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(color_feature)

    ax.set_xlabel(x_feature, fontsize=14)
    ax.set_ylabel(y_feature, fontsize=14)
    ax.set_title(
        f"Activation Energy (Ea) Dependence Plot for {x_feature}",
        fontsize=16,
        fontweight="bold",
    )

    # Match SHAP dependence style – disable grid and use plain axes ticks
    ax.grid(False)

    plt.tight_layout()
    png_path = out_path.with_suffix(".png")
    svg_path = out_path.with_suffix(".svg")
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path, format="svg")
    plt.close(fig)

    print(f"Saved dependence plot → {png_path} (+ .svg)")


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Ea dependence scatter plot from Monte-Carlo predictions CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="mc_ea_predictions.csv",
        help="Path to the Monte-Carlo predictions CSV file.",
    )
    parser.add_argument(
        "--x",
        type=str,
        default="Ash_SiO2",
        help="Feature name for the x-axis.",
    )
    parser.add_argument(
        "--y",
        type=str,
        default="Ea_kJmol",
        help="Feature name for the y-axis.",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="Ash_Na2O",
        help="Feature name used for colouring points.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ea_dependence",
        help="Output directory or filename base (relative to CSV directory).",
    )
    parser.add_argument(
        "--no-data",
        action="store_true",
        help="Skip exporting the underlying scatter data to CSV.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = [col for col in [args.x, args.y, args.color] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s) in CSV: {', '.join(missing)}")

    # Interpret --output ------------------------------------------------------
    out_arg = Path(args.output)
    if out_arg.suffix:  # user included extension ⇒ treat as full path
        base_path = out_arg
        if not base_path.is_absolute():
            base_path = csv_path.parent / base_path
        base_dir = base_path.parent
        file_stem = base_path.stem
    else:
        # Treat as directory name OR file stem depending on presence of '/'
        if "/" in args.output or "\\" in args.output:
            base_dir = (csv_path.parent / out_arg).resolve()
            file_stem = out_arg.name
        else:
            base_dir = (csv_path.parent / args.output).resolve()
            file_stem = args.output

    base_dir.mkdir(parents=True, exist_ok=True)

    out_path = base_dir / file_stem

    # Draw plot and save
    draw_dependence(df, args.x, args.y, args.color, out_path)

    # Export data if requested
    if not args.no_data:
        data_csv = out_path.parent / f"{out_path.stem}_data.csv"
        export_plot_data(df, args.x, args.y, args.color, data_csv)


if __name__ == "__main__":
    main() 