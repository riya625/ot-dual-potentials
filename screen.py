"""Sparse candidate-graph construction from a predicted dual potential u_hat.

Given u_hat (from DualPotentialTransformer) and the cost matrix C for one OT
instance: compute the closed-form c-transform v_hat, the reduced costs r_hat,
and threshold r_hat into a sparse candidate graph (topk per source row, or a
flat epsilon cutoff). See CLAUDE (OT).md's "Sparsity graph construction"
section for the theory (complementary slackness: the true optimal support is
contained in the zero-reduced-cost edges).

Usage as a library:
    v_hat = compute_v_hat(u_hat, C)
    r_hat = compute_reduced_costs(u_hat, v_hat, C)
    mask = build_sparse_graph(r_hat, method="topk", k=10)

`r_hat` is computed once per instance; build_sparse_graph is then cheap to
call repeatedly against that same r_hat with different threshold values
(each call is a single O(n^2) numpy op, no re-solving anything).
"""

import numpy as np


def compute_v_hat(u_hat, C):
    """Closed-form c-transform: v_hat_j = min_i (C[i, j] - u_hat[i]).

    u_hat: (n,) predicted source potentials
    C: (n, m) cost matrix
    returns: (m,) target potentials
    """
    return (C - u_hat[:, None]).min(axis=0)


def compute_reduced_costs(u_hat, v_hat, C):
    """r_hat[i, j] = C[i, j] - u_hat[i] - v_hat[j].

    u_hat: (n,), v_hat: (m,), C: (n, m) -> returns (n, m).
    """
    return C - u_hat[:, None] - v_hat[None, :]


def build_sparse_graph(r_hat, method="topk", k=None, epsilon=None):
    """Threshold r_hat into a sparse candidate graph.

    method="topk": keep the k smallest-r_hat edges per source row (bottom-k
        per row) -- guarantees every source point keeps at least k candidate
        targets, regardless of the absolute scale of r_hat. Says nothing
        about column (target) in-degree -- a column can end up starved,
        which is a common cause of infeasibility on the restricted graph.
    method="topk_bidirectional": row-wise top-k (as above) OR'd with
        column-wise top-k (for each target j, keep the k smallest
        r_hat[:, j] across sources) -- fixes the column-starvation problem
        by construction, at the cost of a denser graph (up to ~2x the edges
        of row-only topk at the same k, minus overlap).
    method="epsilon": keep every edge with r_hat <= epsilon -- a flat cutoff,
        row out-degree can vary (or be zero).

    r_hat: (n, m) reduced-cost matrix.
    returns: (n, m) boolean mask, True = edge kept.
    """
    r_hat = np.asarray(r_hat)
    n_src, n_tgt = r_hat.shape

    if method == "topk":
        if k is None:
            raise ValueError("k is required for method='topk'")
        k = min(k, n_tgt)
        # argpartition: O(n log k) per row, no need to fully sort each row.
        idx = np.argpartition(r_hat, k - 1, axis=1)[:, :k]
        mask = np.zeros_like(r_hat, dtype=bool)
        rows = np.repeat(np.arange(n_src), k)
        mask[rows, idx.ravel()] = True
        return mask

    if method == "topk_bidirectional":
        if k is None:
            raise ValueError("k is required for method='topk_bidirectional'")
        row_mask = build_sparse_graph(r_hat, method="topk", k=k)
        # Column-wise top-k = row-wise top-k of the transpose, transposed back.
        col_mask = build_sparse_graph(r_hat.T, method="topk", k=k).T
        return row_mask | col_mask

    if method == "epsilon":
        if epsilon is None:
            raise ValueError("epsilon is required for method='epsilon'")
        return r_hat <= epsilon

    raise ValueError(
        f"unknown method {method!r}, expected 'topk', 'topk_bidirectional', or 'epsilon'"
    )


if __name__ == "__main__":
    # Smoke test: one test instance, load the trained model, build candidate
    # graphs at a few k values, report sparsity (density / out-degree) and
    # coverage of the true optimal support.
    import argparse

    import ot
    import torch

    from model import DualPotentialTransformer

    parser = argparse.ArgumentParser(description="screen.py smoke test")
    parser.add_argument("--test_path", type=str, default="data/test.npz")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/model.pth")
    args = parser.parse_args()
    TEST_PATH = args.test_path
    CHECKPOINT_PATH = args.checkpoint_path

    data = np.load(TEST_PATH)
    X, Y, pi_star = data["X"][0], data["Y"][0], data["pi_star"][0]
    n_points = X.shape[0]
    input_dim = X.shape[-1]

    model = DualPotentialTransformer(input_dim=input_dim)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        X_t = torch.from_numpy(X).float().unsqueeze(0)
        Y_t = torch.from_numpy(Y).float().unsqueeze(0)
        u_hat = model(X_t, Y_t)[0].numpy()

    C = ot.dist(X, Y, metric="sqeuclidean")
    v_hat = compute_v_hat(u_hat, C)
    r_hat = compute_reduced_costs(u_hat, v_hat, C)

    true_support = pi_star > 1e-8
    n_true_edges = int(true_support.sum())
    print(f"test instance 0: n={n_points}, true support has {n_true_edges} edges "
          f"({n_true_edges / n_points**2:.4%} of all {n_points**2} possible edges)")

    print(f"{'k':>5} {'edges kept':>12} {'density':>10} {'coverage':>10}")
    for k in (3, 5, 10, 20, 50):
        mask = build_sparse_graph(r_hat, method="topk", k=k)
        n_kept = int(mask.sum())
        density = n_kept / mask.size
        coverage = (mask & true_support).sum() / n_true_edges
        print(f"{k:5d} {n_kept:12d} {density:10.4f} {coverage:10.4f}")

    print(f"\n{'epsilon':>10} {'edges kept':>12} {'density':>10} {'coverage':>10}")
    for eps in np.quantile(r_hat, [0.01, 0.02, 0.05, 0.10]):
        mask = build_sparse_graph(r_hat, method="epsilon", epsilon=eps)
        n_kept = int(mask.sum())
        density = n_kept / mask.size
        coverage = (mask & true_support).sum() / n_true_edges
        print(f"{eps:10.3f} {n_kept:12d} {density:10.4f} {coverage:10.4f}")
