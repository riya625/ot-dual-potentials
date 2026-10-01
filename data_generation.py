# GMM point-cloud sampling and OT-instance drawing.

#Produces raw (X, Y) source/target point clouds. `ot_solve.py` consumes these,
#solves each instance exactly, and writes the final dataset files.

# See data_generation_spec_OT_solve.md for all details.


import argparse
import time

import numpy as np

# Fixed GMM parameters (from the spec). Same mixture for source and target,
# UNLESS a decoupled target is requested (see DECOUPLED_TARGET_* below).
# Hardcoded here for reproducibility.
DIM = 5
WEIGHTS = [0.3, 0.4, 0.3]
_BASE_MEANS = [
    [-4, -4, -4, -4, -4],
    [0, 5, 0, 5, 0],
    [5, -3, 5, -3, 5],
]
COVS = [
    (1.0 * np.eye(DIM)).tolist(),
    (1.5 * np.eye(DIM)).tolist(),
    (0.7 * np.eye(DIM)).tolist(),
]

# SEPARATION_SCALE scales the component means toward their weights-weighted
# centroid: MEANS_scaled[i] = CENTROID + SEPARATION_SCALE * (_BASE_MEANS[i] -
# CENTROID). 1.0 reproduces _BASE_MEANS exactly (the original hardcoded
# geometry). <1 pulls the three modes together (more overlap, harder OT);
# >1 pushes them apart (less overlap, easier OT). WEIGHTS and COVS are left
# fixed -- only the mean geometry changes.
SEPARATION_SCALE = 1.0


def compute_centroid(means, weights):
    """Weights-weighted mean of the component means."""
    means = np.asarray(means, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return (weights[:, None] * means).sum(axis=0)


def compute_means(separation_scale=SEPARATION_SCALE, base_means=_BASE_MEANS, weights=WEIGHTS):
    """Scale component means toward their weights-weighted centroid.

    separation_scale=1.0 reproduces `base_means` exactly, by construction:
    centroid + 1.0 * (base_means - centroid) == base_means.
    """
    base = np.asarray(base_means, dtype=float)
    centroid = compute_centroid(base, weights)
    return (centroid + separation_scale * (base - centroid)).tolist()


MEANS = compute_means(SEPARATION_SCALE)  # == _BASE_MEANS at the default scale

# Decoupled source/target: an alternative GMM for the TARGET cloud only, used
# when ot_solve.py is run with --decoupled_target. Different weights (mixing
# proportions reordered/changed) and means (baseline geometry, shifted by a
# constant offset) -- same covariance family (COVS) as the source, since nothing
# asked to vary those too. This is one reasonable choice among many for "a
# different GMM"; flag if you want a different construction.
DECOUPLED_TARGET_WEIGHTS = [0.4, 0.35, 0.25]
DECOUPLED_TARGET_MEAN_OFFSET = [2.0, -1.0, 2.0, -1.0, 2.0]
DECOUPLED_TARGET_MEANS = (
    np.asarray(compute_means(1.0)) + np.asarray(DECOUPLED_TARGET_MEAN_OFFSET)
).tolist()
DECOUPLED_TARGET_COVS = COVS

DEFAULT_N_POINTS = 100
DEFAULT_NUM_INSTANCES = 300  # sanity-check value; scale to ~3000 once confirmed
DEFAULT_SEED = 0


def gmm_params_dict(weights=WEIGHTS, means=MEANS, covs=COVS,
                     target_weights=None, target_means=None, target_covs=None):
    """The mixture parameters as plain Python, for dumping to gmm_params.json.
    If target_* are given (decoupled source/target), includes them too."""
    d = {"weights": weights, "means": means, "covs": covs}
    if target_means is not None:
        d["target_weights"] = target_weights
        d["target_means"] = target_means
        d["target_covs"] = target_covs
    return d


def sample_gmm_cloud(n_points, weights, means, covs, rng):
    """Draw `n_points` points independently from the full mixture.

    Each point independently gets its own component draw (standard mixture
    sampling), as opposed to assigning an entire cloud to one component.

    Returns an array of shape (n_points, d) where d is the coordinate
    dimension inferred from `means`.
    """
    weights = np.asarray(weights, dtype=float)
    means = np.asarray(means, dtype=float)
    covs = np.asarray(covs, dtype=float)
    dim = means.shape[1]

    points = np.empty((n_points, dim), dtype=float)
    for i in range(n_points):
        c = rng.choice(len(weights), p=weights)
        points[i] = rng.multivariate_normal(means[c], covs[c])
    return points


def generate_instance(n_points, weights, means, covs, rng,
                       weights_tgt=None, means_tgt=None, covs_tgt=None):
    """Sample one OT instance: source cloud X and target cloud Y, independently.

    By default the target is drawn from the same (weights, means, covs) as the
    source. Pass weights_tgt/means_tgt/covs_tgt to draw the target from a
    different GMM instead (decoupled source/target).
    """
    X = sample_gmm_cloud(n_points, weights, means, covs, rng)
    w_t = weights if weights_tgt is None else weights_tgt
    m_t = means if means_tgt is None else means_tgt
    c_t = covs if covs_tgt is None else covs_tgt
    Y = sample_gmm_cloud(n_points, w_t, m_t, c_t, rng)
    return X, Y


def generate_dataset(num_instances, n_points, weights, means, covs, seed,
                      weights_tgt=None, means_tgt=None, covs_tgt=None):
    """Generate `num_instances` independent (X, Y) pairs from one seeded rng.

    The same rng is threaded through every instance (no per-instance reseeding)
    so the whole dataset is reproducible from `seed` alone. weights_tgt/
    means_tgt/covs_tgt: optional decoupled target GMM (see generate_instance).
    """
    rng = np.random.default_rng(seed)
    instances = []
    for _ in range(num_instances):
        instances.append(
            generate_instance(n_points, weights, means, covs, rng, weights_tgt, means_tgt, covs_tgt)
        )
    return instances


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity-check GMM data generation.")
    parser.add_argument("--num_instances", type=int, default=DEFAULT_NUM_INSTANCES)
    parser.add_argument("--n_points", type=int, default=DEFAULT_N_POINTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    t0 = time.time()
    data = generate_dataset(
        args.num_instances, args.n_points, WEIGHTS, MEANS, COVS, args.seed
    )
    elapsed = time.time() - t0

    X0, Y0 = data[0]
    print(f"generated {len(data)} instances in {elapsed:.2f}s")
    print(f"per-instance X shape {X0.shape}, Y shape {Y0.shape}")
    print(f"X0 range: [{X0.min():.2f}, {X0.max():.2f}]")
