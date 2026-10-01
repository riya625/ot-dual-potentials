"""Evaluation for the OT dual-potential pipeline.

Part 1 -- potential-regression quality: how well does u_hat match u_star on
the held-out test set (RMSE, R^2, and a scatter plot)?

Part 2 -- downstream screening quality (scaled-down version of
evaluate_py_spec.md's Part 2: correctness only, no timing yet): sweep
candidate-graph thresholds and, per test instance, measure how much of the
true optimal support the candidate graph covers and whether restricting the
exact solve to that graph still recovers the true optimal cost.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ot
import scipy.sparse as sp
import torch
from scipy.optimize import linprog
from torch.utils.data import DataLoader

from model import DualPotentialTransformer
from screen import build_sparse_graph, compute_reduced_costs, compute_v_hat
from train import BATCH_SIZE, CHECKPOINT_PATH, TEST_PATH, PointCloudDataset

RESULTS_DIR = "results"
SCATTER_PATH = os.path.join(RESULTS_DIR, "regression_scatter.png")
SWEEP_PLOT_PATH = os.path.join(RESULTS_DIR, "coverage_correctness_vs_k.png")
BIDIR_COMPARISON_PLOT_PATH = os.path.join(RESULTS_DIR, "topk_row_vs_bidirectional.png")
SCATTER_MAX_POINTS = 5000  # subsample the scatter for a readable plot
SEED = 0

# ============================================================================
# Part 1: potential-regression quality
# ============================================================================


def load_model(checkpoint_path, input_dim, device):
    model = DualPotentialTransformer(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def predict_all(model, loader, device):
    """Run the model over an entire loader; return concatenated (u_hat, u_star)
    numpy arrays of shape (N, n) each, in the loader's (unshuffled) order."""
    u_hats, u_stars = [], []
    with torch.no_grad():
        for X, Y, u_star in loader:
            X, Y = X.to(device), Y.to(device)
            u_hats.append(model(X, Y).cpu().numpy())
            u_stars.append(u_star.numpy())
    return np.concatenate(u_hats, axis=0), np.concatenate(u_stars, axis=0)


def evaluate_regression(u_hat_all, u_star_all, scatter_path, rng):
    """Pooled RMSE/R^2 over every (instance, point), plus a u_hat-vs-u_star
    scatter plot. R^2 = 1 - MSE / Var(u_star), same definition as train.py."""
    mse = float(np.mean((u_hat_all - u_star_all) ** 2))
    rmse = float(np.sqrt(mse))
    r2 = 1.0 - mse / float(np.var(u_star_all))
    print(
        f"[regression] test RMSE={rmse:.4f}  R^2={r2:.4f}  "
        f"({u_hat_all.shape[0]} instances x {u_hat_all.shape[1]} points)"
    )

    flat_hat, flat_star = u_hat_all.ravel(), u_star_all.ravel()
    if flat_hat.size > SCATTER_MAX_POINTS:
        idx = rng.choice(flat_hat.size, size=SCATTER_MAX_POINTS, replace=False)
        flat_hat, flat_star = flat_hat[idx], flat_star[idx]

    lo, hi = float(flat_star.min()), float(flat_star.max())
    plt.figure()
    plt.scatter(flat_star, flat_hat, s=4, alpha=0.3)
    plt.plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--")
    plt.xlabel(r"$u^\star$ (true)")
    plt.ylabel(r"$\hat{u}$ (predicted)")
    plt.title(f"Predicted vs true dual potential (test set)  RMSE={rmse:.3f}  $R^2$={r2:.3f}")
    os.makedirs(os.path.dirname(scatter_path), exist_ok=True)
    plt.savefig(scatter_path)
    print(f"saved regression scatter ({flat_hat.size} of {u_hat_all.size} points) to {scatter_path}")
    return rmse, r2


# ============================================================================
# Part 2: downstream screening quality (coverage + correctness)
# ============================================================================
#
# Restricted exact-solve method: scipy.optimize.linprog on ONLY the kept
# edges as decision variables (row/column marginal equalities, sparse A_eq,
# method="highs") -- NOT POT's network simplex with disallowed edges masked
# to a large cost. We didn't want to assume that trick is clean: forcing
# infeasible mass through a huge-cost edge is still "feasible" by
# construction, so a network-simplex run would always report *some* finite
# cost even when the restricted graph cannot actually carry the true optimal
# plan, silently inflating the correctness numbers. linprog restricted to
# just the kept edges instead reports outright infeasibility
# (`success=False`) when the candidate graph can't satisfy both marginals --
# which happens more than you'd expect (see `n_infeasible` in the summary):
# topk only controls each row's out-degree, not each column's in-degree, so
# a column can be starved, or -- even with every column non-empty -- the
# bipartite subgraph can still fail the transportation-feasibility condition.
#
# Verified below (`_verify_restricted_solver`) that this machinery reproduces
# the already-known exact cost when nothing is restricted (mask = full
# graph), before trusting any of the restricted numbers.

# k=n_points is the full graph (every column kept for every row) -- the
# ceiling of the sweep; included to show correctness saturating at 1.0.
K_VALUES = [3, 5, 10, 20, 30, 50, 75, 100]
BIDIR_K_VALUES = [3, 5, 10, 20, 30, 50]  # for the row-only vs. bidirectional comparison
INCLUDE_EPSILON = False  # epsilon lost the coverage/correctness comparison
                          # at matched density (see prior run); topk-only for now
# Epsilon thresholds are NOT picked from the test set (that would tune the
# threshold on the very data we're measuring correctness on). Instead they're
# quantiles of r_hat pooled over a sample of TRAIN instances -- mirrors
# CLAUDE.md's "tune against coverage on validation data" -- chosen at the
# same quantiles as topk's k/n densities so the two methods are compared at
# matched candidate-graph density.
EPSILON_QUANTILES = [0.03, 0.05, 0.10, 0.20]
EPSILON_TRAIN_PATH = "data/train.npz"
EPSILON_SAMPLE_SIZE = 200
COVERAGE_EDGE_THRESHOLD = 1e-8  # pi_star[i, j] above this = "true support"
CORRECTNESS_REL_TOL = 1e-6


def restricted_exact_solve(C, a, b, mask):
    """Exact transportation LP restricted to `mask`'s edges only.

    Returns (cost, success). success=False (cost=inf) if the restricted
    graph cannot satisfy the marginals at all.
    """
    n_src, n_tgt = C.shape
    src_idx, tgt_idx = np.nonzero(mask)
    n_edges = len(src_idx)
    if n_edges == 0:
        return float("inf"), False

    c = C[src_idx, tgt_idx]
    rows = np.concatenate([src_idx, n_src + tgt_idx])
    cols = np.concatenate([np.arange(n_edges), np.arange(n_edges)])
    A_eq = sp.csr_matrix((np.ones(2 * n_edges), (rows, cols)), shape=(n_src + n_tgt, n_edges))
    b_eq = np.concatenate([a, b])

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not res.success:
        return float("inf"), False
    return float(res.fun), True


def _verify_restricted_solver(X, Y, pi_star):
    """Sanity check requested before trusting the correctness numbers: on the
    unrestricted (full) graph, our linprog machinery must reproduce the cost
    already implied by the saved exact solution (pi_star)."""
    n = X.shape[0]
    C = ot.dist(X, Y, metric="sqeuclidean")
    a = b = np.ones(n) / n
    true_cost = float(np.sum(pi_star * C))
    full_mask = np.ones((n, n), dtype=bool)
    cost, success = restricted_exact_solve(C, a, b, full_mask)
    assert success, "full-graph linprog solve did not converge"
    rel_err = abs(cost - true_cost) / max(true_cost, 1e-12)
    print(f"[verify] full-graph linprog cost={cost:.6f} vs true={true_cost:.6f}  rel_err={rel_err:.2e}")
    assert rel_err < CORRECTNESS_REL_TOL, "restricted-solve machinery disagrees with the known exact solve!"


def compute_epsilon_thresholds(model, device, train_path, quantiles, sample_size, seed):
    """Pick epsilon thresholds as quantiles of the pooled r_hat distribution
    over a random sample of TRAIN instances -- see the note above K_VALUES
    for why train (not test) is used to pick the cutoff."""
    data = np.load(train_path)
    X_all, Y_all = data["X"], data["Y"]
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(X_all), size=min(sample_size, len(X_all)), replace=False)

    all_r = []
    with torch.no_grad():
        for idx in idxs:
            X, Y = X_all[idx], Y_all[idx]
            X_t = torch.from_numpy(X).float().unsqueeze(0).to(device)
            Y_t = torch.from_numpy(Y).float().unsqueeze(0).to(device)
            u_hat = model(X_t, Y_t)[0].cpu().numpy()
            C = ot.dist(X, Y, metric="sqeuclidean")
            v_hat = compute_v_hat(u_hat, C)
            all_r.append(compute_reduced_costs(u_hat, v_hat, C).ravel())
    all_r = np.concatenate(all_r)
    thresholds = np.quantile(all_r, quantiles)
    print(
        f"[epsilon] thresholds from {len(idxs)} train instances: "
        + ", ".join(f"q={q:.2f}->eps={e:.4f}" for q, e in zip(quantiles, thresholds))
    )
    return thresholds


def save_sweep_csv(k_values, results, out_path):
    """Machine-readable copy of a run_screening_sweep table: one row per k."""
    import csv

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "coverage", "correctness", "infeasible_rate", "density"])
        for k in k_values:
            r = results[f"k={k}"]
            writer.writerow([k, r["coverage"], r["correctness"], r["infeasible_rate"], r["density"]])
    print(f"saved sweep CSV to {out_path}")


def run_screening_sweep(X_all, Y_all, pi_star_all, u_hat_all, configs, header="screening sweep summary"):
    """configs: list of (label, method, kwargs) passed to build_sparse_graph,
    e.g. ("k=10", "topk", {"k": 10}) or ("eps=2.66", "epsilon", {"epsilon": 2.66})."""
    n_instances, n_points, _ = X_all.shape
    a = b = np.ones(n_points) / n_points

    coverage_sum = {label: 0.0 for label, _, _ in configs}
    n_correct = {label: 0 for label, _, _ in configs}
    n_infeasible = {label: 0 for label, _, _ in configs}
    density_sum = {label: 0.0 for label, _, _ in configs}

    for idx in range(n_instances):
        X, Y, pi_star, u_hat = X_all[idx], Y_all[idx], pi_star_all[idx], u_hat_all[idx]

        C = ot.dist(X, Y, metric="sqeuclidean")
        v_hat = compute_v_hat(u_hat, C)
        r_hat = compute_reduced_costs(u_hat, v_hat, C)

        true_support = pi_star > COVERAGE_EDGE_THRESHOLD
        n_true_edges = int(true_support.sum())
        true_cost = float(np.sum(pi_star * C))

        for label, method, kwargs in configs:
            mask = build_sparse_graph(r_hat, method=method, **kwargs)
            coverage_sum[label] += (mask & true_support).sum() / n_true_edges
            density_sum[label] += mask.mean()

            cost, success = restricted_exact_solve(C, a, b, mask)
            if not success:
                n_infeasible[label] += 1
                continue
            rel_err = abs(cost - true_cost) / max(true_cost, 1e-12)
            if rel_err <= CORRECTNESS_REL_TOL:
                n_correct[label] += 1

        if (idx + 1) % 100 == 0 or (idx + 1) == n_instances:
            print(f"  screened {idx + 1}/{n_instances} test instances")

    print(f"\n=== {header} ===")
    print(f"{'config':>10} {'mean coverage':>14} {'correctness rate':>18} {'infeasible':>12} {'density':>10}")
    results = {}
    for label, _, _ in configs:
        coverage = coverage_sum[label] / n_instances
        correctness = n_correct[label] / n_instances
        infeasible_rate = n_infeasible[label] / n_instances
        density = density_sum[label] / n_instances
        results[label] = {
            "coverage": coverage,
            "correctness": correctness,
            "infeasible_rate": infeasible_rate,
            "density": density,
        }
        print(f"{label:>10} {coverage:14.4f} {correctness:18.4f} {n_infeasible[label]:12d} {density:10.4f}")
    return results


def plot_screening_sweep(k_values, results, out_path, n_points):
    """coverage / correctness / infeasible-rate vs k, all on one log-x plot."""
    labels = [f"k={k}" for k in k_values]
    coverage = [results[l]["coverage"] for l in labels]
    correctness = [results[l]["correctness"] for l in labels]
    infeasible_rate = [results[l]["infeasible_rate"] for l in labels]

    plt.figure()
    plt.plot(k_values, coverage, "o-", label="coverage (of true support)")
    plt.plot(k_values, correctness, "o-", label="correctness (restricted solve = true optimum)")
    plt.plot(k_values, infeasible_rate, "o--", label="infeasible rate", color="gray")
    plt.xscale("log")
    plt.xticks(k_values, [str(k) for k in k_values])
    plt.xlabel(f"k (candidate edges kept per source row, out of n={n_points})")
    plt.ylabel("rate (fraction of test instances)")
    plt.ylim(-0.05, 1.05)
    plt.title("Sparsity vs. coverage/correctness (topk candidate graph, test set)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    print(f"saved sparsity/coverage/correctness plot to {out_path}")


def _plot_sweep_on_axis(ax, k_values, results, title):
    labels = [f"k={k}" for k in k_values]
    coverage = [results[l]["coverage"] for l in labels]
    correctness = [results[l]["correctness"] for l in labels]
    infeasible_rate = [results[l]["infeasible_rate"] for l in labels]
    density = [results[l]["density"] for l in labels]

    ax.plot(k_values, coverage, "o-", label="coverage")
    ax.plot(k_values, correctness, "o-", label="correctness")
    ax.plot(k_values, infeasible_rate, "o--", label="infeasible rate", color="gray")
    ax.plot(k_values, density, "o:", label="graph density", color="green")
    ax.set_xscale("log")
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_xlabel("k")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_method_comparison(k_values, results_row, results_bidir, out_path, n_points):
    """Row-only topk vs. topk_bidirectional, side by side on matched axes so
    the two are directly comparable (same k values, same y-scale, same
    metrics: coverage, correctness, infeasible rate, graph density)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    _plot_sweep_on_axis(ax1, k_values, results_row, "topk (row-only)")
    _plot_sweep_on_axis(ax2, k_values, results_bidir, "topk_bidirectional (row ∪ column)")
    ax1.set_ylabel("rate (fraction of test instances) / density")
    ax2.legend(loc="lower right")
    fig.suptitle(f"Row-only vs. bidirectional topk candidate graph (n={n_points}, test set)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    print(f"saved row-only vs bidirectional comparison plot to {out_path}")


def main(test_path=TEST_PATH, checkpoint_path=CHECKPOINT_PATH, results_dir=RESULTS_DIR,
         k_values=K_VALUES, bidir_k_values=BIDIR_K_VALUES, seed=SEED, include_epsilon=INCLUDE_EPSILON,
         run_row_only_full_sweep=True):
    scatter_path = os.path.join(results_dir, "regression_scatter.png")
    sweep_plot_path = os.path.join(results_dir, "coverage_correctness_vs_k.png")
    bidir_comparison_plot_path = os.path.join(results_dir, "topk_row_vs_bidirectional.png")
    row_sweep_csv = os.path.join(results_dir, "row_topk_sweep.csv")
    bidir_sweep_csv = os.path.join(results_dir, "bidirectional_topk_sweep.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)

    test_ds = PointCloudDataset(test_path)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    input_dim = test_ds.X.shape[-1]

    model = load_model(checkpoint_path, input_dim, device)
    u_hat_all, u_star_all = predict_all(model, test_loader, device)

    print("=" * 60)
    print("Part 1: potential-regression quality")
    print("=" * 60)
    evaluate_regression(u_hat_all, u_star_all, scatter_path, rng)

    data = np.load(test_path)
    X_all, Y_all, pi_star_all = data["X"], data["Y"], data["pi_star"]
    _verify_restricted_solver(X_all[0], Y_all[0], pi_star_all[0])

    if run_row_only_full_sweep:
        print()
        print("=" * 60)
        print("Part 2: downstream screening quality (correctness only, no timing)")
        print("=" * 60)
        configs = [(f"k={k}", "topk", {"k": k}) for k in k_values]
        if include_epsilon:
            eps_thresholds = compute_epsilon_thresholds(
                model, device, EPSILON_TRAIN_PATH, EPSILON_QUANTILES, EPSILON_SAMPLE_SIZE, seed
            )
            configs += [(f"eps={eps:.2f}", "epsilon", {"epsilon": eps}) for eps in eps_thresholds]
        results = run_screening_sweep(X_all, Y_all, pi_star_all, u_hat_all, configs)
        if not include_epsilon:
            plot_screening_sweep(k_values, results, sweep_plot_path, n_points=X_all.shape[1])

    print()
    print("=" * 60)
    print("Part 2b: row-only topk vs. topk_bidirectional")
    print("=" * 60)
    row_configs = [(f"k={k}", "topk", {"k": k}) for k in bidir_k_values]
    bidir_configs = [(f"k={k}", "topk_bidirectional", {"k": k}) for k in bidir_k_values]
    results_row = run_screening_sweep(
        X_all, Y_all, pi_star_all, u_hat_all, row_configs, header="topk (row-only)"
    )
    results_bidir = run_screening_sweep(
        X_all, Y_all, pi_star_all, u_hat_all, bidir_configs, header="topk_bidirectional"
    )
    plot_method_comparison(
        bidir_k_values, results_row, results_bidir, bidir_comparison_plot_path, n_points=X_all.shape[1]
    )
    save_sweep_csv(bidir_k_values, results_row, row_sweep_csv)
    save_sweep_csv(bidir_k_values, results_bidir, bidir_sweep_csv)
    return results_row, results_bidir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_path", type=str, default=TEST_PATH)
    parser.add_argument("--checkpoint_path", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bidir_k_values", type=str, default=None,
                         help="comma-separated k values for the row-only/bidirectional "
                              "comparison sweep, e.g. '3,5,10,20,30,50'. Default: same as baseline.")
    parser.add_argument("--skip_row_only_full_sweep", action="store_true",
                         help="skip Part 2's full row-only k=3..100 sweep (Part 2b's row-only "
                              "comparison still runs) -- saves time when only the bidirectional "
                              "numbers are needed.")
    args = parser.parse_args()

    kwargs = dict(
        test_path=args.test_path, checkpoint_path=args.checkpoint_path,
        results_dir=args.results_dir, seed=args.seed,
        run_row_only_full_sweep=not args.skip_row_only_full_sweep,
    )
    if args.bidir_k_values is not None:
        kwargs["bidir_k_values"] = [int(k) for k in args.bidir_k_values.split(",")]
    main(**kwargs)
