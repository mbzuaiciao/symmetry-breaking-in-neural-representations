"""Integration tests for experiment modules."""

import os

from experiments.exp_a_seed_dependence import run_experiment_a
from experiments.exp_b_training_trajectories import run_experiment_b
from experiments.exp_c_external_field import run_experiment_c
from experiments.exp_d_representation_alignment import run_experiment_d
from experiments.exp_e_architecture_comparison import run_experiment_e
from experiments.exp_f_data_asymmetry import run_experiment_f
from experiments.exp_g_width_sweep import run_experiment_g
from experiments.exp_h_regularization import run_experiment_h


def test_smoke_exp_a(tmp_path):
    res = run_experiment_a(output_dir=str(tmp_path), assets_dir=str(tmp_path), smoke=True)
    assert len(res["order_parameters"]) == 5
    assert os.path.exists(tmp_path / "exp_a_seed_dependence.json")
    assert os.path.exists(tmp_path / "exp_a_order_parameter_hist.png")


def test_smoke_exp_b(tmp_path):
    res = run_experiment_b(output_dir=str(tmp_path), assets_dir=str(tmp_path), smoke=True)
    assert res["num_seeds"] == 3
    assert os.path.exists(tmp_path / "exp_b_training_trajectories.json")


def test_smoke_exp_c(tmp_path):
    res = run_experiment_c(output_dir=str(tmp_path), assets_dir=str(tmp_path), smoke=True)
    assert len(res["epsilons"]) == 3
    assert os.path.exists(tmp_path / "exp_c_external_field.json")


def test_smoke_exp_d(tmp_path):
    res = run_experiment_d(output_dir=str(tmp_path), assets_dir=str(tmp_path), smoke=True)
    assert res["num_models"] == 4
    assert os.path.exists(tmp_path / "exp_d_representation_alignment.json")


def test_smoke_exp_e(tmp_path):
    res = run_experiment_e(output_dir=str(tmp_path), assets_dir=str(tmp_path), smoke=True)
    assert len(res["sample_sizes"]) == 2
    assert os.path.exists(tmp_path / "exp_e_architecture_comparison.json")


def test_smoke_exp_f(tmp_path):
    res = run_experiment_f(output_dir=str(tmp_path), assets_dir=str(tmp_path), smoke=True)
    assert len(res["deltas"]) == 3
    assert os.path.exists(tmp_path / "exp_f_data_asymmetry.json")


def test_smoke_exp_g(tmp_path):
    res = run_experiment_g(output_dir=str(tmp_path), assets_dir=str(tmp_path), smoke=True)
    assert len(res["widths"]) == 3
    assert os.path.exists(tmp_path / "exp_g_width_sweep.json")


def test_smoke_exp_h(tmp_path):
    res = run_experiment_h(output_dir=str(tmp_path), assets_dir=str(tmp_path), smoke=True)
    assert len(res["weight_decays"]) == 3
    assert os.path.exists(tmp_path / "exp_h_regularization.json")
