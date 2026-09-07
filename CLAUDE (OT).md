# Learning Dual Potentials to Accelerate Repeated Discrete Optimal Transport

## Project goal

We repeatedly solve discrete OT problems between finite samples drawn from the same
underlying (but unknown) continuous source and target distributions. Instead of
learning to predict the transport plan directly, we learn the **Kantorovich dual
potential** `u(x)`, and use it to predict a **superset of the optimal support** so
that an *exact* LP solver can run on a much smaller candidate graph. Correctness is
guaranteed by the exact solver; the learned model only affects speed.

Full math background and the argument for learning `u` rather than the Brenier
potential `φ` is in `docs/background.md` (to be filled in from prior discussion —
see "Background" section below for the short version).

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
  learned), then `r̂_ij = c_ij - û_i - v̂_j`, then threshold to build a sparse
  candidate graph, then solve the exact LP restricted to that graph.

## Data generation

- **Distribution**: 2D Gaussian mixture, 3 components, well-separated for
  visualization/debugging. Suggested defaults (adjust if desired):
  ```python
  weights = [0.3, 0.4, 0.3]
  means   = [[-4, -4], [0, 5], [5, -3]]
  covs    = [[[1.0, 0.3], [0.3, 1.0]],
             [[1.5, -0.4], [-0.4, 0.8]],
             [[0.7, 0.0], [0.0, 0.7]]]
  ```
  Same mixture used for both source and target (α = β = this GMM), unless later
  experiments want source ≠ target.
- **Sampling per instance**: for EACH OT instance, draw source cloud
  `X = {x_1,...,x_n}` and target cloud `Y = {y_1,...,y_n}` **independently**, each
  point independently drawn from the full mixture (standard GMM sampling — each
  point's component chosen i.i.d. according to `weights`, then drawn from that
  component's Gaussian). Do NOT assign an entire cloud to a single component
  (this was an issue in the prior implementation — see "Known issues" below).
- **Point cloud size**: `n = 100` points per cloud (both source and target).
- **Number of instances**: a few thousand (e.g. 2000–5000) independent
  `(X, Y)` draws. Each is a fully separate OT problem — not pairwise
  combinations of a smaller pool of clouds.
- **Train/test split**: 80/20, split at the instance level (not within a cloud).

## Exact OT solve (per instance)

- Cost: squared Euclidean, `c_ij = ||x_i - y_j||^2` (matches the quadratic-cost /
  Brenier's theorem connection used in the background).
- Solver: POT (`ot.solve_sample(..., metric='sqeuclidean', method='emd')`), which
  returns both the transport plan and the dual potentials (`res.potentials`).
- **Gauge fixing**: dual potentials are only defined up to a shared additive
  constant (`u → u+c, v → v-c` leaves feasibility unchanged). Before saving/using
  `u*` as a training target, normalize per instance (e.g. subtract `mean(u*)`, or
  fix `u*_1 = 0`). Skipping this makes the regression target inconsistent across
  instances for reasons unrelated to model quality — do not skip it.
- Save per instance: `X, Y, u* (normalized), v* (normalized), π*` (needed later for
  computing true support / coverage metrics).

## Model: transformer for learning û

- **Input**: source cloud `X` (n×2) and target cloud `Y` (n×2).
- **Required symmetry**: output must be permutation-*equivariant* in the source
  indices (relabeling source points relabels the output the same way) and
  permutation-*invariant* in the target indices (relabeling target points doesn't
  change the output at all). This is why a CNN over the plan matrix was wrong in
  the prior version — matrix row/column position isn't meaningful spatial
  structure here.
- **Architecture**:
  1. Linear embedding of each point (2D coordinate → d-dim embedding) for both
     clouds.
  2. Self-attention within source cloud, self-attention within target cloud.
  3. Cross-attention: source points attend to target points (this is the
     mechanism most directly analogous to what reduced costs measure — pairwise
     source/target compatibility).
  4. Pointwise MLP decoder head → scalar `û_i` per source point.
  5. No positional encoding (point order is arbitrary, not sequential).
- **Loss**: MSE between `û_i` and normalized `u*_i`, per instance, averaged over
  the batch.
- **Training**: standard train/test split as above; log train and test loss per
  epoch.

## Sparsity graph construction (inference time, per test instance)

1. Run the trained model on `(X, Y)` to get `û_i` for all source points.
2. Compute `v̂_j = min_i (c_ij - û_i)` (closed form, not learned).
3. Compute `r̂_ij = c_ij - û_i - v̂_j` for all `i, j`.
4. Threshold: keep edges with the smallest `r̂_ij` — either bottom-`k` per row or
   below a cutoff `ε`. This is a tunable hyperparameter, not a theoretical
   constant — tune it against coverage on validation data (see Evaluation).
5. Build the resulting sparse candidate graph.

## Exact LP solve on the reduced graph

Solve the discrete OT LP restricted to the candidate graph's edges only (e.g. via
a sparse/restricted network simplex or by masking disallowed edges with `+∞`
cost in POT). Compare against the full exact solve for evaluation.

## Evaluation

Two distinct things to evaluate — both wanted per the professor's ask for
"results and graphs":

1. **Model regression quality** (does û learn?): train/test loss curves for the
   `u` regression over training epochs.
2. **Downstream pipeline quality** (does screening actually work?), computed on
   held-out test instances where the true support is known from the exact solve:
   - **Coverage**: fraction of true-support edges contained in the candidate
     graph, as a function of threshold/sparsity level.
   - **Sparsity vs. coverage** curve: the key plot — how much can the graph be
     shrunk while still reliably covering the true support.
   - **Speed**: exact LP solve time on the screened graph vs. on the full graph.
   - **Correctness check**: does the restricted solve recover the same optimal
     cost/plan as the full solve when coverage holds.

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

## Suggested repo structure

```
data_generation.py   # GMM sampling, OT instance drawing
ot_solve.py           # exact POT solve + gauge normalization + save
model.py               # transformer architecture
train.py                # training loop, loss curves
screen.py               # reduced-cost computation, thresholding, candidate graph
evaluate.py             # coverage, sparsity, speed, correctness metrics + plots
```
