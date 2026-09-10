import time 
import numpy as np 
import ot 

def generate_OT_pairs(fname='test_sample.npy'):
    X = np.load(fname)
    n_dist, n_points, n_dim = X.shape
    i, j = np.triu_indices(n_dist, k=1)

    sources_block = X[i]
    targets_block = X[j]

    ot_distances = np.zeros(len(i))
    ot_plans = np.zeros((len(i), n_points, n_points))
    dual_source = np.zeros(((len(i)), n_points))
    dual_target = np.zeros(((len(i)), n_points))

    a = np.ones(n_points) / n_points
    b = np.ones(n_points) / n_points

    start = time.time()
    for k in range(len(i)):
        src = sources_block[k]
        tgt = targets_block[k]
        res = ot.solve_sample(src, tgt, a, b, metric='sqeuclidean', method='emd')
        ot_plans[k, :, :] = res.plan
        alpha, beta = res.potentials

        # Normalize the additive-constant ambiguity: anchor alpha to zero mean,
        # shift beta oppositely so alpha[i] + beta[j] sums are preserved.
        shift = alpha.mean()
        alpha = alpha - shift
        beta = beta + shift

        dual_source[k, :] = alpha
        dual_target[k, :] = beta
        ot_distances[k] = np.sqrt(res.value)
        if k % 1000 == 0:
            print(f"Iter {k}: Current run time: {time.time() - start}")

    print(f"total run time: {time.time() - start}")
    return (sources_block, targets_block, dual_source, dual_target, ot_distances, ot_plans, n_points)
