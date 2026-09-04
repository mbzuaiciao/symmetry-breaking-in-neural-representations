"""Master orchestrator to run all experiments and generate all figures & results."""

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.exp_a_seed_dependence import run_experiment_a
from experiments.exp_b_training_trajectories import run_experiment_b
from experiments.exp_c_external_field import run_experiment_c
from experiments.exp_d_representation_alignment import run_experiment_d
from experiments.exp_e_architecture_comparison import run_experiment_e
from experiments.exp_f_data_asymmetry import run_experiment_f
from experiments.exp_g_width_sweep import run_experiment_g
from experiments.exp_h_regularization import run_experiment_h


def main():
    parser = argparse.ArgumentParser(description="Run all symmetry breaking experiments")
    parser.add_argument("--smoke", action="store_true", help="Run in fast smoke test mode")
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Directory for JSON results"
    )
    parser.add_argument("--assets-dir", type=str, default="assets", help="Directory for figures")
    args = parser.parse_args()

    mode_str = "SMOKE TEST" if args.smoke else "FULL EXPERIMENTAL SUITE"
    print("==================================================")
    print(f" Starting {mode_str}")
    print("==================================================")

    start_time = time.time()

    # Exp A
    print("\n>>> Running Exp A (Seed Dependence)...")
    run_experiment_a(output_dir=args.output_dir, assets_dir=args.assets_dir, smoke=args.smoke)

    # Exp B
    print("\n>>> Running Exp B (Training Trajectories)...")
    run_experiment_b(output_dir=args.output_dir, assets_dir=args.assets_dir, smoke=args.smoke)

    # Exp C
    print("\n>>> Running Exp C (External Perturbation Field)...")
    run_experiment_c(output_dir=args.output_dir, assets_dir=args.assets_dir, smoke=args.smoke)

    # Exp D
    print("\n>>> Running Exp D (Representation Alignment & Procrustes)...")
    run_experiment_d(output_dir=args.output_dir, assets_dir=args.assets_dir, smoke=args.smoke)

    # Exp E
    print("\n>>> Running Exp E (Architecture Comparison)...")
    run_experiment_e(output_dir=args.output_dir, assets_dir=args.assets_dir, smoke=args.smoke)

    # Exp F
    print("\n>>> Running Exp F (Data Asymmetry)...")
    run_experiment_f(output_dir=args.output_dir, assets_dir=args.assets_dir, smoke=args.smoke)

    # Exp G
    print("\n>>> Running Exp G (Width Sweep)...")
    run_experiment_g(output_dir=args.output_dir, assets_dir=args.assets_dir, smoke=args.smoke)

    # Exp H
    print("\n>>> Running Exp H (Regularization & Weight Decay)...")
    run_experiment_h(output_dir=args.output_dir, assets_dir=args.assets_dir, smoke=args.smoke)

    elapsed = time.time() - start_time
    print("\n==================================================")
    print(f" Suite Completed Successfully in {elapsed:.1f}s")
    print(f" Results saved to: {args.output_dir}/")
    print(f" Figures saved to: {args.assets_dir}/")
    print("==================================================")


if __name__ == "__main__":
    main()
