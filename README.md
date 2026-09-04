# Symmetry Breaking in Neural Representations

A rigorous, intuitive tutorial laboratory and experimental framework exploring symmetry, representation geometry, and spontaneous symmetry breaking in neural networks—designed to support research initiatives and NeurReps-style workshop submissions.

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![Package Manager](https://img.shields.io/badge/uv-fast%20python-purple)](https://astral.sh/uv)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked](https://img.shields.io/badge/type%20checker-pyright-green)](https://github.com/microsoft/pyright)
[![Tests](https://img.shields.io/badge/tests-50%2B%20passed-brightgreen)](https://pytest.org/)

---

## 🎯 Motivation & Core Research Question

Deep neural networks do not simply map inputs to outputs; they construct high-dimensional geometric representations of data. When learning tasks possess intrinsic symmetries (reflection, rotation, translation, permutation), how do internal representations organize?

> ### The Central Research Question
> **When does a neural network preserve the symmetry of a task, and when does optimization select an asymmetric internal representation?**

```text
Symmetry
   ↓
Group Actions & Representations
   ↓
Invariance vs. Equivariance
   ↓
Representation Geometry (Covariance, Gram, Eigenspectra)
   ↓
Spontaneous Symmetry Breaking & Order Parameters
   ↓
Training Dynamics & Bifurcation Transitions
   ↓
Representation Alignment (Orthogonal Procrustes & Linear CKA)
   ↓
Controlled Perturbation Experiments
   ↓
NeurReps Research Program
```

---

## 🔬 Key Mathematical Metrics

All mathematical metrics are implemented from first principles in `src/symmetry_reps/` with unit tests for invariance properties:

1. **Roy & Vetterli Effective Rank**:
   $$\text{erank}(X) = \exp(H(p)) = \exp\left(-\sum_{i=1}^r p_i \ln p_i\right), \quad p_i = \frac{\sigma_i}{\sum \sigma_j}$$
2. **Participation Ratio (PR)**:
   $$\text{PR}(\Sigma) = \frac{(\text{Tr}\,\Sigma)^2}{\text{Tr}(\Sigma^2)} = \frac{\left(\sum_{i=1}^d \lambda_i\right)^2}{\sum_{i=1}^d \lambda_i^2}$$
3. **Linear Centered Kernel Alignment (CKA)**:
   $$\text{CKA}(X_1, X_2) = \frac{\|\tilde{X}_1^T \tilde{X}_2\|_F^2}{\|\tilde{X}_1^T \tilde{X}_1\|_F \|\tilde{X}_2^T \tilde{X}_2\|_F}$$
   *Invariances*: Orthogonal rotations $O(d)$ and isotropic positive scalings.
4. **Orthogonal Procrustes Disparity**:
   $$\min_{R \in O(d)} \|X_1 R - X_2\|_F^2 \implies R^* = U V^T \quad \text{where } X_1^T X_2 = U \Sigma V^T$$
5. **Symmetry-Breaking Order Parameter**:
   $$m = \frac{1}{N} \sum_{i=1}^N y_i \langle h(x_i), v_{\text{sym}} \rangle$$
6. **Neural Collapse Metrics (NC1, NC2, NC3)**:
   - NC1 (Variability Collapse): $\text{Tr}(\Sigma_W \Sigma_B^\dagger) / C$
   - NC2 (Simplex ETF Angle Equiangularity): $\text{Var}(\langle \tilde{\mu}_c, \tilde{\mu}_{c'} \rangle)$
   - NC3 (Duality): $\frac{1}{C}\sum (1 - \cos(w_c, \tilde{\mu}_c))$

---

## ⚡ Quickstart with `uv`

Dependency management is handled exclusively via `uv`.

```bash
# 1. Clone repository
git clone https://github.com/mbzuaiciao/symmetry-breaking-in-neural-representations.git
cd symmetry-breaking-in-neural-representations

# 2. Sync dependencies
uv sync --extra dev

# 3. Run all unit and integration tests (52 tests)
uv run pytest -v

# 4. Run full check (format, lint, typecheck, tests)
make check

# 5. Run all research experiments and generate figures
make figures
```

### Makefile Targets

| Target | Description |
| :--- | :--- |
| `make sync` | Install dependencies via `uv` |
| `make check` | Run formatting check, linting, typechecking, and full test suite |
| `make test` | Run pytest suite |
| `make lint` | Run ruff linter |
| `make format` | Run ruff code formatter |
| `make typecheck`| Run pyright type checker |
| `make smoke` | Run fast smoke test across all 8 experiments |
| `make figures` | Run full experimental suite and save publication figures to `assets/` |

---

## 🧪 Central Experimental Suite

The repository contains 8 reproducible, laptop-friendly experiments in `experiments/`:

- **[Exp A: Seed Dependence](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/experiments/exp_a_seed_dependence.py)**: Demonstrates that individual models break symmetry into polarized states while the ensemble of runs remains symmetric ($\langle m \rangle \approx 0$).
- **[Exp B: Training Dynamics](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/experiments/exp_b_training_trajectories.py)**: Tracks step-by-step evolution of loss, order parameter, effective rank, and CKA.
- **[Exp C: External Symmetry-Breaking Field](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/experiments/exp_c_external_field.py)**: Sweeps external bias $\epsilon \in [0, 10^{-1}]$ and measures orientation selection probability $P(m > 0)$.
- **[Exp D: Representation Alignment](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/experiments/exp_d_representation_alignment.py)**: Proves that independently trained models with opposite orientations are pairwise isometric under Orthogonal Procrustes alignment.
- **[Exp E: Architecture Comparison](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/experiments/exp_e_architecture_comparison.py)**: Compares unconstrained MLPs vs symmetry-aware architectures across training sample sizes.
- **[Exp F: Ground-Truth Data Asymmetry](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/experiments/exp_f_data_asymmetry.py)**: Studies representation phase transition as data distribution asymmetry $\delta$ increases.
- **[Exp G: Network Width Sweep](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/experiments/exp_g_width_sweep.py)**: Analyzes how hidden dimension $W \in [4, 128]$ affects order-parameter variance and rank compression.
- **[Exp H: Regularization and Weight Decay](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/experiments/exp_h_regularization.py)**: Quantifies how weight decay $\lambda$ controls covariance anisotropy and representation norm.

---

## 📚 Comprehensive Tutorial Curriculum

The `tutorials/` directory provides 41 in-depth Markdown tutorials across 12 parts:

### Part I — Foundations
- [00 — What This Repository Is About](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/00_what_this_repository_is_about.md)
- [01 — Representations as Geometry](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/01_representations_as_geometry.md)
- [02 — Symmetry from First Principles](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/02_symmetry_from_first_principles.md)
- [03 — Groups and Group Actions](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/03_groups_and_group_actions.md)
- [04 — Symmetry in Neural Networks](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/04_symmetry_in_neural_networks.md)

### Part II — Symmetry Breaking
- [05 — What Is Symmetry Breaking?](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/05_what_is_symmetry_breaking.md)
- [06 — A Minimal Neural Symmetry-Breaking Experiment](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/06_a_minimal_neural_symmetry_breaking_experiment.md)
- [07 — Order Parameters for Neural Representations](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/07_order_parameters_for_neural_representations.md)

### Part III — Representation Geometry
- [08 — PCA, Eigenspectra, and Anisotropy](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/08_pca_eigenspectra_and_anisotropy.md)
- [09 — Representation Similarity (Linear CKA)](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/09_representation_similarity.md)
- [10 — Orthogonal Procrustes Alignment](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/10_procrustes_alignment.md)
- [11 — CCA and Subspace Similarity](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/11_cca_and_subspace_similarity.md)
- [12 — Intrinsic Dimension and Representation Collapse](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/12_intrinsic_dimension.md)

### Part IV — Training Dynamics
- [13 — Watching Representations Form (Checkpointing)](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/13_watching_representations_form.md)
- [14 — Symmetry Breaking as a Dynamical Process](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/14_symmetry_breaking_as_a_dynamical_process.md)
- [15 — Initialization and Symmetry](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/15_initialization_and_symmetry.md)
- [16 — Explicit Perturbations (External Fields)](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/16_explicit_perturbations.md)

### Part V — Architecture and Equivariance
- [17 — Hard-Coded Equivariance vs. Learned Symmetry](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/17_hard_coded_equivariance_vs_learned_symmetry.md)
- [18 — When Symmetry Constraints Help](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/18_when_symmetry_constraints_help.md)
- [19 — When Symmetry Constraints Hurt](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/19_when_symmetry_constraints_hurt.md)

### Part VI — Loss Landscapes and Parameter Symmetry
- [20 — Hidden-Unit Permutation Symmetry](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/20_hidden_unit_permutation_symmetry.md)
- [21 — Scale Symmetries in ReLU Networks](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/21_scale_symmetries.md)
- [22 — Flat Directions and Loss Hessians](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/22_flat_directions_and_hessians.md)
- [23 — Goldstone Modes: Physics Analogy & Limits](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/23_goldstone_mode_intuition.md)

### Part VII — Neural Collapse and Emergent Geometry
- [24 — Neural Collapse: Emergent Simplex ETF Geometry](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/24_neural_collapse.md)
- [25 — Is Neural Collapse a Symmetry Phenomenon?](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/25_is_neural_collapse_a_symmetry_phenomenon.md)

### Part VIII — World Models and Dynamics
- [26 — Symmetry in Dynamical Systems](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/26_symmetry_in_dynamical_systems.md)
- [27 — Learning Equivariant Latent Dynamics](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/27_learning_equivariant_latent_dynamics.md)
- [28 — Symmetry Breaking in World Models](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/28_symmetry_breaking_in_world_models.md)

### Part IX — Topology and Manifolds
- [29 — Neural Manifolds](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/29_neural_manifolds.md)
- [30 — Topology of Representations (Betti Numbers & PH)](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/30_topology_of_representations.md)
- [31 — Symmetry and Manifold Structure (Orbits)](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/31_symmetry_and_manifold_structure.md)

### Part X — Neuroscience Connection
- [32 — Representational Similarity Analysis (RSA)](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/32_representational_similarity_analysis.md)
- [33 — Grid Cells and Geometric Neural Codes](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/33_grid_cells_and_geometric_codes.md)
- [34 — Neural Manifolds and Population Codes](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/34_neural_manifolds_and_population_codes.md)

### Part XI — Language and Modern Representation Learning
- [35 — Geometry of Embeddings](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/35_geometry_of_embeddings.md)
- [36 — The Linear Representation Hypothesis](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/36_linear_representation_hypotheses.md)
- [37 — Superposition, Polysemanticity, and Discrete Symmetries](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/37_superposition_and_symmetry.md)

### Part XII — Research Methodology
- [38 — How to Study Representation Geometry Scientifically](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/38_how_to_study_representation_geometry_scientifically.md)
- [39 — Common Failure Modes in Representation Geometry](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/39_common_failure_modes.md)
- [40 — From Tutorial to Research Paper: NeurReps Pipeline](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/tutorials/40_from_tutorial_to_research_paper.md)

---

## 📝 Research Notes

Detailed workshop paper proposals and literature surveys in `notes/`:
- [`research_questions.md`](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/notes/research_questions.md): Core research program and falsification hypotheses.
- [`neurreps_paper_ideas.md`](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/notes/neurreps_paper_ideas.md): 4 structured NeurReps workshop paper proposals with ratings and target venues.
- [`experiment_backlog.md`](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/notes/experiment_backlog.md): Planned follow-up experiments.
- [`literature_map.md`](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/notes/literature_map.md): Bibliography covering GDL, CKA, Neural Collapse, and Statistical Physics.
- [`open_questions.md`](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/notes/open_questions.md): Deep open theoretical questions.
- [`results_interpretation.md`](file:///Users/xq/Documents/GitHub/symmetry-breaking-in-neural-representations/notes/results_interpretation.md): Detailed scientific synthesis of empirical observations.

---

## 📄 License

MIT License.
