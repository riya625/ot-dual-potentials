# Learning Dual Potentials to Accelerate Repeated Discrete Optimal Transport

## Project goal

We repeatedly solve discrete OT problems between finite samples drawn from the same
underlying (but unknown) continuous source and target distributions. Instead of
learning to predict the transport plan directly, we learn the **Kantorovich dual
potential** `u(x)`, and use it to predict a **superset of the optimal support** so
that an *exact* LP solver can run on a much smaller candidate graph. Correctness is
guaranteed by the exact solver; the learned model only affects speed.

> This file is kept in sync with the actual code (last synced after the
> separation-scale sweep experiment). Where the original plan and the final
> implementation diverge, this file describes what was **actually built** and
> notes the deviation. Per-file implementation specs (interface, reference
> code, rationale for each design choice) live in the sibling
> `*_spec*.md` files — `model_py_spec.md`, `data_generation_spec_OT_solve.md`,
> `train_py_spec.md` — and are the authoritative detail; this file is the
> project-level summary.

## Background (short version)

- Discrete OT primal: transport plan `π_ij ≥ 0` minimizing `Σ c_ij π_ij` subject to
  row/column marginal constraints.
- Dual: potentials `u_i, v_j` maximizing `Σ u_i α_i + Σ v_j β_j` subject to
  `u_i + v_j ≤ c_ij` for all `i,j`. Solved *simultaneously* with the primal by the
  network simplex method — the dual is a byproduct of the same solve, not a
  separate computation.
- Reduced cost: `r_ij = c_ij - u_i - v_j ≥ 0`. Complementary slackness:
  `π*_ij > 0 ⟹ r_ij = 0`. So the true optimal support is contained in the set of
  zero-reduced-cost edges — this is the theoretical basis for screening.
- We learn `u` (not `φ`) because: (a) it interfaces directly with the solver via
  `r_ij`, (b) it only requires predicting scalar values, not derivatives, (c) it's
  more robust to sampling noise across repeated draws from the same distributions.
- Once `û_i` is predicted: `v̂_j = min_i(c_ij - û_i)` (closed-form c-transform, not
  learned) — note this makes `(û, v̂)` a dual-**feasible** pair by construction
  (r̂_ij ≥ 0 always), just not necessarily optimal — then `r̂_ij = c_ij - û_i - v̂_j`,
  then threshold to build a sparse candidate graph, then solve the exact LP
  restricted to that graph.

## Data generation (`data_generation.py`)

- **Distribution**: Gaussian mixture, 3 components. Ended up at **5D** (`DIM = 5`),
  not the original 2D suggestion — the model, pipeline, and all specs were
  generalized to an `input_dim` parameter early on, and the dataset settled on 5D.
  Base geometry (`_BASE_MEANS`), fixed:
  ```python
  DIM = 5
  WEIGHTS  = [0.3, 0.4, 0.3]
  _BASE_MEANS = [
      [-4, -4, -4, -4, -4],
      [ 0,  5,  0,  5,  0],
      [ 5, -3,  5, -3,  5],
  ]
  COVS = [1.0*eye(5), 1.5*eye(5), 0.7*eye(5)]   # isotropic per component
  ```
  Same mixture used for both source and target by default (α = β), **unless**
  `--decoupled_target` is passed (see "Separation-scale / distribution-shift
  experiments" below).
- **SEPARATION_SCALE** (added later, not in the original plan): scales the
  three component means toward their weights-weighted centroid —
  `MEANS_scaled[i] = CENTROID + SEPARATION_SCALE * (_BASE_MEANS[i] - CENTROID)`.
  `SEPARATION_SCALE = 1.0` reproduces `_BASE_MEANS` exactly (verified: max abs
  diff 0.0; pairwise mean distances √210, √245, √203 ≈ 14.49, 15.65, 14.25).
  `<1` pulls the modes together (more overlap, harder OT); `>1` pushes them
  apart. `WEIGHTS` and `COVS` are left fixed when scaling. Exposed via
  `compute_means(scale)` and `ot_solve.py --separation_scale`.
- **Sampling per instance**: for EACH OT instance, draw source cloud
  `X = {x_1,...,x_n}` and target cloud `Y = {y_1,...,y_n}` **independently**, each
  point independently drawn from the full mixture (`sample_gmm_cloud`: each
  point's component chosen i.i.d. via `rng.choice(..., p=weights)`, then drawn
  from that component's Gaussian). Point clouds are never assigned to a single
  component (see "Known issues" below — this was the prior implementation's bug).
- **Point cloud size**: `n = 100` points per cloud (both source and target).
- **Number of instances**: sanity runs at `num_instances=300` (240/60 split);
  the real results below are at **`num_instances=3000`** (2400/600 split).
- **Train/test split**: 80/20 (`TRAIN_FRAC = 0.8` in `ot_solve.py`), a fixed
  slice of the seeded instance list — not re-randomized at save time.
- CLI: `python ot_solve.py --num_instances N --n_points 100 --seed 0 --out_dir DIR
  [--separation_scale S] [--decoupled_target]` (also generates the data via
  `data_generation.generate_dataset`, which `ot_solve.py` calls).

## Exact OT solve (`ot_solve.py`, per instance)

- Cost: squared Euclidean, `c_ij = ||x_i - y_j||^2`.
- Solver: `ot.solve_sample(X, Y, a, b, metric="sqeuclidean")` — **no `method=` arg**.
  The originally-planned `method='emd'` is not accepted by the installed POT
  version (0.9.7); omitting `method`/`reg` already runs the exact network-simplex
  solver and returns `res.potentials` as the dual, which is what's needed.
- **Gauge fixing** (`normalize_potentials`): `u_norm = u - mean(u)`,
  `v_norm = v + mean(u)` — leaves `u_i + v_j` unchanged, forces `mean(u_norm) = 0`
  per instance. Checked (`_check_normalization`) on the first 5 instances: mean
  ≈ 0 and `u_norm[i] + v_norm[j] ≤ C[i,j] + 1e-6` spot-checked on random pairs.
- **Saved arrays** (`np.savez_compressed`):
  - `data/train.npz`: `X, Y (N,n,DIM)`, `u_star (N,n)` — deliberately **no**
    `pi_star`/`v_star` (training never needs them; a full `(N,n,n)` plan array
    would be ~large for no benefit).
  - `data/test.npz`: adds `v_star (N,n)` and `pi_star (N,n,n)` — needed by
    `evaluate.py` for true support / coverage / correctness.
  - `data/gmm_params.json`: weights/means/covs (+ separation_scale,
    decoupled_target, target_* if decoupled) for reproducibility.
- CLI: `python ot_solve.py --num_instances 3000 --out_dir data` (see full args above).

## Model: transformer for learning û (`model.py`)

**Architecture actually implemented is simpler than the original plan** — one
joint encoder-only transformer over the concatenated source+target sequence,
not separate self-attention (source, target) + cross-attention blocks. The
joint sequence gives source↔source, target↔target, *and* source↔target
attention automatically in one mechanism; separate blocks would have been a
more complex, unneeded alternative. Full rationale in `model_py_spec.md`.

- `DualPotentialTransformer(input_dim=5, d_model=64, nhead=4, num_layers=4,
  dim_ff=256, dropout=0.1)`.
- **Embedding**: one `nn.Linear(input_dim, d_model)` for both clouds, plus a
  learned 2-row `nn.Embedding` (source tag / target tag) added on top so the
  shared linear layer can still tell the clouds apart.
- **Sequence**: `[source tokens (n) ; target tokens (n)]` → one
  `nn.TransformerEncoder` (no causal masking, **no positional encoding** —
  required for permutation-equivariance in source order / invariance in
  target order).
- **Read-out**: only the first `n` (source) token outputs go through the head
  (`Linear → ReLU → Linear(1)`); target tokens only inform source tokens via
  attention, never predicted at.
- **Normalization** (added after the first training run got stuck at the
  predict-the-mean baseline — raw ±8-range coordinates and `std(u_star)≈24`
  made the problem badly conditioned): three non-trainable buffers,
  `coord_mean`/`coord_std` (`(input_dim,)`) and `u_std` (scalar), set from the
  **training set only** via `model.set_normalization(...)`. `forward`
  standardizes `X`/`Y` before embedding and multiplies the head's output by
  `u_std`, so `forward` still returns `û` in raw `u_star` units. Buffers
  default to mean 0/std 1 (identity) and travel in the `state_dict`, so
  inference code (`screen.py`, `evaluate.py`) needs nothing beyond
  `load_state_dict`.
- **Sanity checks** (in `model.py`'s `__main__`, run before any training):
  shape (`(8,100,input_dim)→(8,100)`), gradient flow (every param has
  `.grad`), source permutation-**equivariance**, target permutation-
  **invariance** (`atol=1e-5`, `model.eval()` to kill dropout noise). All pass.

## Training (`train.py`)

- Config constants: `SEED=0`, `BATCH_SIZE=32`, `LEARNING_RATE=1e-3`,
  `GRAD_CLIP_NORM=1.0`, `NUM_EPOCHS=100`.
- **Gradient clipping was necessary, not optional** — an early 5D run with no
  seed and no clipping collapsed mid-training (output went constant, R²→0).
  Fixing the seed and adding `clip_grad_norm_` made training reproducible and
  stable.
- **Both MSE and R² tracked**, train and test, every epoch, via
  `evaluate_metrics` (one clean `model.eval()` pass; `R² = 1 - MSE/Var(u_star)`,
  pooled over every (instance, point), so R²=0 means "no better than the mean").
  `train_one_epoch` only optimizes; it doesn't return a running loss — both
  curves come from the same clean-pass function, so they're directly
  comparable.
- Outputs per run: checkpoint (`state_dict`, includes the normalization
  buffers), `loss_curve.png`, `r2_curve.png`, and `metrics.csv` (one row per
  epoch: `epoch, train_mse, test_mse, train_r2, test_r2` — the exact numbers
  each plot is drawn from).
- All paths + `num_epochs`/`seed` are CLI-overridable
  (`--train_path/--test_path/--checkpoint_path/--loss_plot_path/--r2_plot_path/
  --metrics_path/--num_epochs/--seed`), defaulting to the baseline paths —
  this is what let later experiments (the separation-scale sweep) run without
  ever touching the baseline's data/checkpoint/results.

**Baseline result** (5D, N=3000, 100 epochs): **test R² = 0.929**, RMSE ≈ 18.8
(u_star std ≈ 70). Train/test curves track closely (no overfitting); still
visibly noisy epoch-to-epoch (small-batch, `LR=1e-3`).

## Sparsity graph construction (`screen.py`)

- `compute_v_hat(u_hat, C)`: `v̂_j = min_i(C[i,j] - û_i)` — closed form.
- `compute_reduced_costs(u_hat, v_hat, C)`: `r̂ = C - û[:,None] - v̂[None,:]`.
- `build_sparse_graph(r_hat, method, k=None, epsilon=None)` → boolean mask,
  three methods:
  - `"topk"`: bottom-`k` `r̂` per **source row**. Guarantees row out-degree `k`,
    says nothing about column in-degree — a column can be starved of edges
    entirely, or (even with every column non-empty) the bipartite subgraph can
    still fail the transportation-feasibility condition. Empirically common at
    small `k`.
  - `"topk_bidirectional"`: row-wise top-k **OR** column-wise top-k (top-k of
    `r_hat.T`, transposed back). Fixes column starvation by construction, at
    higher density than row-only for the same `k`. **This is the method that
    actually works well** — see Evaluation results below.
  - `"epsilon"`: flat cutoff `r̂ ≤ ε`. **Tried and empirically worse than topk
    at matched density** — a flat global cutoff doesn't guarantee every row
    gets a fair share of candidates, so hard rows can be shut out entirely
    while easy rows waste budget on edges they don't need. Kept in the code
    (`INCLUDE_EPSILON` toggle in `evaluate.py`) but not the recommended method.
- Each `build_sparse_graph` call is a single O(n²) numpy op — cheap to call
  repeatedly against the same `r̂` at many threshold values, which is exactly
  what the evaluation sweeps below do.
- Script-mode smoke test (`python screen.py [--test_path ...] [--checkpoint_path ...]`):
  one test instance, prints density/coverage across a few `k` and `ε` values.

## Exact LP solve on the reduced graph (in `evaluate.py`)

**Deviated from the original plan of "restricted network simplex or +∞-cost
masking in POT"** — used `scipy.optimize.linprog` (sparse `A_eq`,
`method="highs"`) with **only the kept edges as decision variables**, not
POT's network simplex with disallowed edges masked to a large cost. Reasoning
(and this mattered — see below): forcing infeasible mass through a huge-cost
edge is still "feasible" by construction, so simplex would always report
*some* finite cost even when the candidate graph genuinely can't carry the
true optimal plan, silently inflating the correctness numbers. `linprog`
restricted to just the kept edges instead reports outright infeasibility
(`success=False`) when the marginals can't be satisfied at all —
**and this happens far more than expected** (see infeasibility numbers below):
`topk` only controls row out-degree, and even a bidirectional graph with every
row/column non-empty can still fail the transportation-feasibility condition.

Verified before trusting any restricted number: on the full (unrestricted)
graph, this `linprog` machinery reproduces the already-known exact cost from
`pi_star` (`_verify_restricted_solver`, `rel_err ~1e-8`, checked on every run).

## Evaluation (`evaluate.py`)

Two parts, matching the two things worth evaluating:

**Part 1 — regression quality** (does `û` learn?): loads the checkpoint +
`test.npz`, predicts `û` over the whole test set, reports pooled RMSE/R², and
saves a `û`-vs-`u_star` scatter (subsampled to ≤5000 points for a readable
plot) to `regression_scatter.png`.

**Part 2 — downstream screening quality** (scaled-down version of the
original plan's Part 2: correctness only, **no wall-clock timing yet**):
sweeps candidate-graph thresholds and, per test instance:
- **Coverage**: fraction of true-support edges (`pi_star > 1e-8`) present in
  the candidate graph.
- **Correctness**: does the `linprog`-restricted exact solve match the true
  cost (`Σ pi_star·C`) within `1e-6` relative tolerance? (Separately from
  outright **infeasibility** — the restricted graph sometimes can't satisfy
  the marginals at all, which is tracked as its own outcome, not folded into
  "incorrect.")
- **Density**: fraction of all `n²` edges kept.

Sweeps both `"topk"` (`K_VALUES = [3,5,10,20,30,50,75,100]`, `k=100`=full
graph, the ceiling) and `"topk_bidirectional"` (`BIDIR_K_VALUES`, default
`[3,5,10,20,30,50]`, widened to `[3,5,...,100]` for the sweep experiments
below), and plots coverage/correctness/infeasible-rate/density vs `k` (log-x).
Everything is CLI-overridable (`--test_path/--checkpoint_path/--results_dir/
--bidir_k_values/--skip_row_only_full_sweep`), which is what lets the same
script evaluate any config's checkpoint without touching the baseline's.

### Baseline results (5D, N=3000, seed 0)

| k | row-only correctness | row-only infeasible | bidirectional correctness | bidirectional infeasible |
|---|---|---|---|---|
| 10 | 0.022 | 571/600 | 0.233 | 254/600 |
| 20 | 0.085 | 484/600 | 0.640 | 131/600 |
| 30 | 0.497 | 102/600 | 0.877 | 26/600 |
| 35 | — | — | **0.965** | 0/600 |
| 50 | 0.977 | 0/600 | 1.000 | 0/600 |
| 100 (full graph) | 1.000 | 0/600 | 1.000 | 0/600 |

**Bidirectional topk clearly wins**: at every matched `k`, higher coverage,
higher correctness, far fewer infeasible instances (row-only needs `k≈75-100`
to fully saturate; bidirectional gets there by `k≈45-50`, and first crosses
0.95 correctness at `k=35`, density≈0.39). This makes sense given the
mechanism above — bidirectional is specifically the fix for row-only's
column-starvation failure mode. `epsilon` (tried once at matched density
against row-only topk) lost on every metric and wasn't pursued further.

## Separation-scale / distribution-shift experiments (`run_separation_sweep.py`)

Beyond the original plan: once the baseline pipeline worked end-to-end, ran a
controlled sweep varying the GMM geometry, to see how separation between
mixture components (i.e. how hard the OT problem is) affects both regression
quality and how much candidate-graph density screening needs. Driver:
`run_separation_sweep.py` — full `data-gen → train → screen → evaluate` cycle
per config, each isolated under `data/<label>/`, `checkpoints/<label>/`,
`results/<label>/` so nothing touches the baseline. Full per-stage logs in
`sweep_logs/`.

Configs (all N=3000, n=100, DIM=5, seed=0, 100 epochs):

| config | separation_scale | source=target GMM? | test R² | smallest k for bidir. correctness≥0.95 | density at that k |
|---|---|---|---|---|---|
| sep_0.15 | 0.15 | yes | 0.867 | 10 | 0.128 |
| sep_0.3 | 0.3 | yes | 0.901 | 10 | 0.125 |
| sep_0.6 | 0.6 | yes | 0.943 | 25 | 0.289 |
| **baseline** | 1.0 | yes | 0.929 | 35 | 0.394 |
| sep_2.0 | 2.0 | yes | 0.943 | 40 | 0.464 |
| decoupled_sep_1.0 | 1.0 | **no** (diff. weights + shifted means) | 0.962 | 35 | 0.393 |

**Findings**:
- **Required screening density is cleanly monotonic in separation scale**:
  tighter cluster overlap (0.15, 0.3) needs *less* candidate-graph density for
  correctness (k=10, density≈0.13); wider separation (2.0) needs *more*
  (k=40, density≈0.46). Working hypothesis (not independently verified beyond
  this one sweep): at low separation the mixture is nearly one blob, so
  there's less within-cluster ambiguity for the model to get wrong; at high
  separation, more plausible same-cluster candidates compete for each true
  edge, so a wider net is needed.
- **Decoupled source/target** (different weights `[0.4,0.35,0.25]` vs.
  `[0.3,0.4,0.3]`, means shifted by `[2,-1,2,-1,2]`, same covariance family):
  the model fit it *better* (R²=0.962, best of all 6 configs) but needed
  essentially the *same* screening density as baseline (k=35, density≈0.39
  either way) — decoupling `α ≠ β` didn't break anything or change the
  sparsification story, at least for this choice of "how different."
- All 5 new configs' full-graph `linprog` verification passed
  (`rel_err < 1e-7`); baseline's own data/checkpoint/results were confirmed
  byte-identical (MD5) after every stage of this sweep — never touched.

## What's not done yet

- **Wall-clock timing** (restricted vs. full LP solve speed) — explicitly out
  of scope for the current evaluation; only correctness has been measured so
  far.
- Nothing has been committed to git for the separation-scale sweep outputs
  (data/checkpoints/results/logs) — everything is saved locally under
  `data/`, `checkpoints/`, `results/`, `sweep_logs/` but uncommitted, pending
  a decision on what to keep in version control (the raw `.npz`/`.pth` files
  are `.gitignore`d regardless; the `.py` changes and result plots/CSVs are
  the candidates for a future commit).

## Known issues in the prior implementation (do not repeat)

- ResNet over the transport plan matrix treated as a grayscale image — wrong
  inductive bias (no real spatial locality; output was a single global
  embedding, not per-point scalars).
- Training targets were all-zero placeholders — no real learning signal existed.
- Data generation had an unresolved dimension mismatch (hardcoded GMM params
  were 2D; the data file used was named as if 5D) and was never actually wired
  up end-to-end in the script.
- Each point cloud was assigned to a single mixture component rather than
  sampled from the full mixture, and OT instances were formed from arbitrary
  pairwise combinations of a smaller pool of clouds rather than fresh
  independent draws per instance.
- Dual potentials were extracted and saved but never used as training targets.

## Repo structure (actual)

```
data_generation.py         # GMM sampling (5D, SEPARATION_SCALE, decoupled target), OT instance drawing
ot_solve.py                 # exact POT solve + gauge normalization + save; CLI for scale/decoupled/out_dir
model.py                     # DualPotentialTransformer: joint-sequence encoder + normalization buffers
train.py                      # training loop: MSE + R^2, checkpoint + 2 plots + metrics.csv; CLI paths
screen.py                      # compute_v_hat, compute_reduced_costs, build_sparse_graph (topk / topk_bidirectional / epsilon)
evaluate.py                     # Part 1 regression (RMSE/R^2/scatter) + Part 2 screening sweep (coverage/correctness/density)
run_separation_sweep.py          # driver: data-gen->train->screen->evaluate per GMM-geometry config, unattended

model_py_spec.md                 # model.py implementation spec (interface, reference impl, rationale)
data_generation_spec_OT_solve.md # data_generation.py + ot_solve.py spec
train_py_spec.md                 # train.py spec

data/            # train.npz/test.npz (baseline) + data/<label>/ per sweep config; gmm_params.json
checkpoints/     # model.pth (baseline) + checkpoints/<label>/model.pth per sweep config
results/         # baseline plots/CSVs + results/<label>/ per sweep config + results/baseline_fine/ (finer re-eval)
sweep_logs/       # full per-stage stdout logs + status.json for the separation-scale sweep
```
