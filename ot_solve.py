#Exact OT solve + gauge normalization + dataset save.

#Entry point for the data pipeline: generates raw (X, Y) clouds via
#`data_generation.generate_dataset`, solves each instance exactly with POT,
#gauge-fixes the dual potentials, splits 80/20, and writes
#`data/train.npz` + `data/test.npz`.

#Run the sanity-check batch first:

#    python ot_solve.py --num_instances 300

#then scale up (e.g. --num_instances 3000) once the shapes/timings check out.


import argparse
import json
import os
import time

import numpy as np
import ot

from data_generation import (
    COVS,
    DECOUPLED_TARGET_COVS,
    DECOUPLED_TARGET_MEANS,
    DECOUPLED_TARGET_WEIGHTS,
    DEFAULT_N_POINTS,
    DEFAULT_NUM_INSTANCES,
    DEFAULT_SEED,
    SEPARATION_SCALE,
    WEIGHTS,
    compute_means,
    generate_dataset,
    gmm_params_dict,
)

TRAIN_FRAC = 0.8


def solve_instance(X, Y):
    """Exact squared-Euclidean OT solve. Returns u, v, pi, cost."""
    a = np.ones(len(X)) / len(X)
    b = np.ones(len(Y)) / len(Y)
    # Exact EMD: solve_sample with no reg / no method argument runs the exact
    # network-simplex solver and returns the dual potentials as a byproduct.
    res = ot.solve_sample(X, Y, a, b, metric="sqeuclidean")
    u, v = res.potentials
    return {
        "u": np.asarray(u),
        "v": np.asarray(v),
        "pi": np.asarray(res.plan),
        "cost": float(res.value),
    }


def normalize_potentials(u, v):
    """Gauge-fixing: shift so mean(u_norm) == 0, leaving u_i + v_j unchanged."""
    c = u.mean()
    return u - c, v + c


def _check_normalization(u_norm, v_norm, X, Y, rng):
    """Verify the gauge shift preserved dual feasibility (no sign error)."""
    assert np.allclose(u_norm.mean(), 0.0), u_norm.mean()
    C = ot.dist(X, Y, metric="sqeuclidean")
    n, m = C.shape
    for _ in range(20):
        i = int(rng.integers(n))
        j = int(rng.integers(m))
        assert u_norm[i] + v_norm[j] <= C[i, j] + 1e-6, (i, j, u_norm[i] + v_norm[j], C[i, j])


def build_and_save_dataset(num_instances, n_points, seed, out_dir,
                            separation_scale=SEPARATION_SCALE, decoupled_target=False):
    os.makedirs(out_dir, exist_ok=True)

    means = compute_means(separation_scale)
    weights_tgt = means_tgt = covs_tgt = None
    if decoupled_target:
        weights_tgt, means_tgt, covs_tgt = (
            DECOUPLED_TARGET_WEIGHTS, DECOUPLED_TARGET_MEANS, DECOUPLED_TARGET_COVS,
        )

    with open(os.path.join(out_dir, "gmm_params.json"), "w") as f:
        json.dump(
            gmm_params_dict(WEIGHTS, means, COVS, weights_tgt, means_tgt, covs_tgt)
            | {"separation_scale": separation_scale, "decoupled_target": decoupled_target},
            f, indent=2,
        )

    print(
        f"generating {num_instances} instances (n_points={n_points}, seed={seed}, "
        f"separation_scale={separation_scale}, decoupled_target={decoupled_target})..."
    )
    t0 = time.time()
    instances = generate_dataset(
        num_instances, n_points, WEIGHTS, means, COVS, seed,
        weights_tgt=weights_tgt, means_tgt=means_tgt, covs_tgt=covs_tgt,
    )
    print(f"  generation done in {time.time() - t0:.2f}s")

    check_rng = np.random.default_rng(seed + 1)

    dim = instances[0][0].shape[1]  # coordinate dimension (2D, 5D, ...)
    X_all = np.empty((num_instances, n_points, dim), dtype=np.float32)
    Y_all = np.empty((num_instances, n_points, dim), dtype=np.float32)
    u_all = np.empty((num_instances, n_points), dtype=np.float32)
    v_all = np.empty((num_instances, n_points), dtype=np.float32)
    pi_all = np.empty((num_instances, n_points, n_points), dtype=np.float32)

    t0 = time.time()
    for k, (X, Y) in enumerate(instances):
        sol = solve_instance(X, Y)
        u_norm, v_norm = normalize_potentials(sol["u"], sol["v"])
        if k < 5:
            _check_normalization(u_norm, v_norm, X, Y, check_rng)

        X_all[k] = X
        Y_all[k] = Y
        u_all[k] = u_norm
        v_all[k] = v_norm
        pi_all[k] = sol["pi"]

        if (k + 1) % 50 == 0 or (k + 1) == num_instances:
            elapsed = time.time() - t0
            print(f"  solved {k + 1}/{num_instances}  ({elapsed:.1f}s, {elapsed / (k + 1):.3f}s/inst)")

    n_train = int(round(num_instances * TRAIN_FRAC))
    print(f"split: {n_train} train / {num_instances - n_train} test")

    train_path = os.path.join(out_dir, "train.npz")
    test_path = os.path.join(out_dir, "test.npz")

    np.savez_compressed(
        train_path,
        X=X_all[:n_train],
        Y=Y_all[:n_train],
        u_star=u_all[:n_train],
        v_star=v_all[:n_train],
    )
    np.savez_compressed(
        test_path,
        X=X_all[n_train:],
        Y=Y_all[n_train:],
        u_star=u_all[n_train:],
        v_star=v_all[n_train:],
        pi_star=pi_all[n_train:],
    )

    total = time.time() - t0
    print(f"wrote {train_path} and {test_path}")
    print(f"total solve+save time: {total:.1f}s  (~{total * 10:.0f}s estimated for 10x instances)")

    for path in (train_path, test_path):
        with np.load(path) as d:
            shapes = {name: d[name].shape for name in d.files}
        print(f"  {os.path.basename(path)}: {shapes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_instances", type=int, default=DEFAULT_NUM_INSTANCES)
    parser.add_argument("--n_points", type=int, default=DEFAULT_N_POINTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--separation_scale", type=float, default=SEPARATION_SCALE,
                         help="scales GMM component means toward their centroid; "
                              "1.0 = baseline geometry")
    parser.add_argument("--decoupled_target", action="store_true",
                         help="draw the target cloud from a different GMM than the source")
    args = parser.parse_args()

    build_and_save_dataset(
        args.num_instances, args.n_points, args.seed, args.out_dir,
        separation_scale=args.separation_scale, decoupled_target=args.decoupled_target,
    )
