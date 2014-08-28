import sys
from toolshed import nopen
import numpy as np
import scipy.stats as ss

choice = np.random.choice


def simulate_cluster(cluster, w, n=None):
    """this is the method from the A-clustering paper
    w = 0 generates random data.
    n tells how many in each group to simulate, so the
    data that is sent in should contain at least 2*N.
    """

    N = len(cluster[0].values)
    if n is None:
        assert N % 2 == 0
        n = N / 2
    assert 2 * n <= N, (n, N)

    n_probes = len(cluster)
    new_data = np.zeros((n_probes, 2 * n))

    # choose a ranomd probe from the set.
    # the values across the cluster will be determined
    # by the values in this randomly chose probe.
    i = choice(range(n_probes))#[0] if n_probes > 1 else 0
    c = cluster[i]

    idxs = np.arange(N)

    # just pull based on the index. so we need to sort the values
    # as well.
    idx_order = np.argsort(c.values)

    ords = np.arange(1, N + 1) / (N + 1.0)
    ords = (1.0 - ords)**w
    h_idxs = choice(idxs, replace=False, p=ords/ords.sum(), size=n)

    idxs = np.setdiff1d(idxs, h_idxs, assume_unique=True)
    idxs.sort()

    ords = np.arange(1, N + 1 - n) / (N + 1.0 - n)
    assert ords.shape == idxs.shape
    ords = (ords)**w
    l_idxs = choice(idxs, replace=False, p=ords/ords.sum(), size=n)

    assert len(np.intersect1d(h_idxs, l_idxs)) == 0
    for j in range(n_probes):
        tmph = cluster[j].values[idx_order][h_idxs]
        tmpl = cluster[j].values[idx_order][l_idxs]
        cluster[j].values[:n] = tmph
        cluster[j].values[n:] = tmpl
    return cluster

def gen_arma(n_probes=20000, n_patients=80, corr=0.2, df=2, scale=0.2):
    # these parameters are taken from the Bump Hunting Paper
    from statsmodels.tsa.arima_process import ArmaProcess
    sigma = 0.5
    rvs = ss.norm(df, loc=0.05 / sigma, scale=scale).rvs
    corr = -abs(corr)
    return np.column_stack([
        sigma * ArmaProcess([1, corr], [1])
                           .generate_sample(n_probes=n_probes, distrvs=rvs)
                           for i in range(n_patients)])

