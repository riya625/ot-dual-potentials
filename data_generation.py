# GMM point-cloud sampling and OT-instance drawing.

#Produces raw (X, Y) source/target point clouds. `ot_solve.py` consumes these,
#solves each instance exactly, and writes the final dataset files.

# See data_generation_spec_OT_solve.md for all details.


import argparse
import time

import numpy as np

# Fixed GMM parameters (from the spec). Same mixture for source and target.
# Hardcoded here for reproducibility.
DIM = 5
WEIGHTS = [0.3, 0.4, 0.3]
MEANS = [
    [-4, -4, -4, -4, -4],
    [0, 5, 0, 5, 0],
    [5, -3, 5, -3, 5],
]
COVS = [
    (1.0 * np.eye(DIM)).tolist(),
    (1.5 * np.eye(DIM)).tolist(),
    (0.7 * np.eye(DIM)).tolist(),
]

DEFAULT_N_POINTS = 100
DEFAULT_NUM_INSTANCES = 300  # sanity-check value; scale to ~3000 once confirmed
DEFAULT_SEED = 0


def gmm_params_dict():
    """The mixture parameters as plain Python, for dumping to gmm_params.json."""
    return {"weights": WEIGHTS, "means": MEANS, "covs": COVS}


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


def generate_instance(n_points, weights, means, covs, rng):
    """Sample one OT instance: source cloud X and target cloud Y, independently."""
    X = sample_gmm_cloud(n_points, weights, means, covs, rng)
    Y = sample_gmm_cloud(n_points, weights, means, covs, rng)
    return X, Y


def generate_dataset(num_instances, n_points, weights, means, covs, seed):
    """Generate `num_instances` independent (X, Y) pairs from one seeded rng.

    The same rng is threaded through every instance (no per-instance reseeding)
    so the whole dataset is reproducible from `seed` alone.
    """
    rng = np.random.default_rng(seed)
    instances = []
    for _ in range(num_instances):
        instances.append(generate_instance(n_points, weights, means, covs, rng))
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
