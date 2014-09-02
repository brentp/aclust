"""
An example of how to use the output from aclust.

For each cluster, we get N probes. Since those will be correlated within
each individual, we use GEE's from a pull request on statsmodels.

We could also use, for example, a mixed-effect model, e.g., in lme4 syntax:

    methylation ~ asthma + age + (1|sample_id)

to allow a random intercept for each sample (where each sample will have
a number of measurements equal to the number of probes in a given cluster.
"""

import sys
from aclust import mclust
import toolshed as ts
from itertools import starmap, izip
import time

import pandas as pd
import numpy as np
import scipy.stats as ss
import patsy

from scipy.stats import norm
from numpy.linalg import cholesky as chol

# TODO: evaluate GLS vs OLS
from statsmodels.api import GEE, GLM, MixedLM, RLM, GLS, OLS
from statsmodels.genmod.dependence_structures import Exchangeable
from statsmodels.genmod.families import Gaussian

def one_cluster(formula, methylation, covs, coef, robust=True):
    """used when we have a "cluster" with 1 probe."""
    c = covs.copy()
    c['methylation'] = methylation
    res = (RLM if robust else GLS).from_formula(formula, data=c).fit()
    return get_ptc(res, coef)

def gee_cluster(formula, methylation, covs, coef, cov_struct=Exchangeable(),
        family=Gaussian()):
    cov_rep = pd.concat((covs for i in range(len(methylation))))
    nr, nc = methylation.shape
    cov_rep['CpG'] = np.repeat(['CpG_%i' % i for i in range(methylation.shape[0])],
                        methylation.shape[1])
    cov_rep['methylation'] = np.concatenate(methylation)

    res = GEE.from_formula(formula, groups=cov_rep['id'], data=cov_rep, cov_struct=cov_struct).fit()
    return get_ptc(res, coef)

def mixed_model_cluster(formula, methylation, covs, coef):
    """TODO."""
    covs['id'] = ['id_%i' % i for i in range(len(covs))]
    cov_rep = pd.concat((covs for i in range(len(methylation))))
    nr, nc = methylation.shape
    cov_rep['CpG'] = np.repeat(['CpG_%i' % i for i in range(methylation.shape[0])],
                        methylation.shape[1])
    cov_rep['methylation'] = np.concatenate(methylation)
    #cov2 = cov_rep[['id', 'CpG', 'methylation', 'age', 'gender', 'asthma']].to_csv('/tmp/m.csv', index=False)

    res = MixedLM.from_formula(formula, groups='id', data=cov_rep).fit()
    #res = res.fit() #free=(np.ones(res.k_fe), np.eye(res.k_re)))
    return get_ptc(res, coef)

def get_ptc(fit, coef):
    idx = [i for i, par in enumerate(fit.model.exog_names)
                       if par.startswith(coef)]
    assert len(idx) == 1, ("too many params like", coef)
    return {'p': fit.pvalues[idx[0]],
            't': fit.tvalues[idx[0]],
            'coef': fit.params[idx[0]]}


def stouffer_liptak(pvals, sigma):
    qvals = norm.isf(pvals).reshape(len(pvals), 1)
    C = np.asmatrix(chol(sigma)).I
    qvals = C * qvals
    Cp = qvals.sum() / np.sqrt(len(qvals))
    return norm.sf(Cp)

def _combine_cluster(formula, methylations, covs, coef, robust=False):
    """function called by z-score and liptak to get pvalues"""
    res = [(RLM if robust else GLS).from_formula(formula, covs).fit()
        for methylation in methylations]

    idx = [i for i, par in enumerate(res[0].model.exog_names)
                   if par.startswith(coef)][0]
    pvals = np.array([r.pvalues[idx] for r in res], dtype=np.float64)
    pvals[pvals == 1] = 1.0 - 9e-16
    return dict(t=np.array([r.tvalues[idx] for r in res]),
                coef=np.array([r.params[idx] for r in res]),
                p=pvals,
                corr=np.abs(ss.spearmanr(methylations.T)[0]))

def liptak_cluster(formula, methylations, covs, coef, robust=True):
    r = _combine_cluster(formula, methylations, covs, coef, robust=robust)
    r['p'] = stouffer_liptak(r['p'], r['corr'])
    r['t'], r['coef'] = r['t'].mean(), r['coef'].mean()
    return r

def zscore_cluster(formula, methylations, covs, coef, robust=False):
    r = _combine_cluster(formula, methylations, covs, coef)
    z, L = np.mean(norm.isf(r['p'])), len(r['p'])
    sz = 1.0 / L * np.sqrt(L + 2 * np.tril(r['corr'], k=-1).sum())
    r['p'] = norm.sf(z/sz)
    r['t'], r['coef'] = r['t'].mean(), r['coef'].mean()
    return r

def wrapper(model_fn, model_str, cluster, clin_df, coef):
    """wrap the user-defined functions to return everything we expect and
    to call just GLS when there is a single probe."""
    t = time()
    if len(cluster) > 1:
        r = model_fn(model_str, np.array([c.values for c in cluster]), clin_df, coef)
    else:
        r = one_cluster(model_str, cluster[0].values, clin_df, coef)
    r['time'] = time.time() - t
    r['chrom'] = cluster[0].chrom
    r['start'] = cluster[0].position
    r['end'] = cluster[-1].position
    r['n_probes'] = len(cluster)
    r['probes'] = ",".join(c.spos for c in cluster)
    r['var'] = coef
    return r


# function for comparing with bump_cluster
# takes the return value of _combine_cluster and returns a single numeric value
def coef_sum(c, cutoff=0.015):
    coefs = c['coef']
    return sum(min(0, c + cutoff) if c < 0 else max(0, c - cutoff) for c in coefs)

# function for comparing with bump_cluster
def t_sum(c, cutoff=2):
    coefs = c['t']
    return sum(min(0, c + cutoff) if c < 0 else max(0, c - cutoff) for c in coefs)

# function for comparing with bump_cluster
def coef_t_prod(coefs):
    return np.median([coefs['t'][i] * coefs['coef'][i]
                        for i in range(len(coefs['coef']))])

def bump_cluster(model_str, methylations, covs, coef, nsims=1000,
        value_fn=coef_sum, robust=False):
    orig = _combine_cluster(model_str, methylations, covs, coef, robust=robust)
    obs_coef = value_fn(orig)

    reduced_residuals, reduced_fitted = [], []

    # get the reduced residuals and models so we can shuffle
    for i, methylation in enumerate(methylations):
        y, X = patsy.dmatrices(model_str, covs, return_type='dataframe')
        idxs = [par for par in X.columns if par.startswith(coef)]
        assert len(idxs) == 1, ('too many coefficents like', coef)
        X.pop(idxs[0])
        fitr = (RLM if robust else GLS)(y, X).fit()

        reduced_residuals.append(np.array(fitr.resid))
        reduced_fitted.append(np.array(fitr.fittedvalues))

    ngt, idxs = 0, np.arange(len(methylations[0]))

    for isim in range(nsims):
        np.random.shuffle(idxs)

        fakem = np.array([rf + rr[idxs] for rf, rr in izip(reduced_fitted,
            reduced_residuals)])
        assert fakem.shape == methylations.shape

        sim = _combine_cluster(model_str, fakem, covs, coef, robust=robust)
        ccut = value_fn(sim)
        ngt += abs(ccut) > abs(obs_coef)
        # sequential monte-carlo.
        if ngt > 5: break

    p = (1.0 + ngt) / (2.0 + isim) # extra 1 in denom for 0-index
    orig['p'] = p
    orig['coef'], orig['t'] = orig['coef'].mean(), orig['t'].mean()
    return orig


def model_clusters(clust_iter, clin_df, model_str, coef, model_fn=gee_cluster):
    for r in ts.pmap(wrapper, ((model_fn, model_str, cluster, clin_df, coef)
                    for cluster in clust_iter)):
        yield r


class Feature(object):
    __slots__ = "chrom position values spos".split()

    def __init__(self, chrom, pos, values):
        self.chrom, self.position, self.values = chrom, pos, np.array(values)
        self.spos = "%s:%i" % (chrom, pos)

    def distance(self, other):
        if self.chrom != other.chrom: return sys.maxint
        return self.position - other.position

    def is_correlated(self, other):
        rho, p = ss.spearmanr(self.values, other.values)
        return rho > 0.7

    def __repr__(self):
        return "Feature({spos})".format(spos=self.spos)

    def __cmp__(self, other):
        return cmp(self.chrom, other.chrom) or cmp(self.position,
                                                   other.position)


def evaluate_method(clust_iter, df, formula, coef, model_fn, n_real, n_fake):

    from simulate import simulate_cluster
    cluster_iter = clust_iter

    clusters = model_clusters(clust_iter, df, formula, coef,
                              model_fn=model_fn)

    trues = []
    tot_time = 0
    for i, c in enumerate(clusters):
        if i == n_real: break
        tot_time += c['time']
        trues.append(c['p'])

    cluster_iter2 = (simulate_cluster(c, w=0) for c in clust_iter)
    df[coef] = [1] * (len(df)/2) + [0] * (len(df)/2)
    clusters = model_clusters(cluster_iter2, df, formula, coef,
                              model_fn=model_fn)

    falses = []
    for i, c in enumerate(clusters):
        tot_time += c['time']
        if i == n_fake: break
        falses.append(c['p'])

    r = dict(method=model_fn.func_name, n_real_tests=n_real,
             n_fake_tests=n_fake, formula=formula, time=tot_time)

    # find number less than each alpha
    for e in range(8):
        v = 10**-e
        r['true_%i' % e] = sum(t <= v for t in trues)
        r['false_%i' % e] = sum(f <= v for f in falses)
    return r

if __name__ == "__main__":

    def feature_gen(fname):
        for i, toks in enumerate(ts.reader(fname, header=False)):
            if i == 0: continue
            chrom, pos = toks[0].split(":")
            yield Feature(chrom, int(pos), map(float, toks[1:]))

    #fmt = "{chrom}\t{start}\t{end}\t{n_probes}\t{p:5g}\t{t:.4f}\t{coef:.4f}\t{var}"
    #print ts.fmt2header(fmt)

    clust_iter = (c for c in mclust(feature_gen(sys.argv[1]),
                                    max_dist=400, max_skip=1) if len(c) > 2)


    df = pd.read_table('meth.clin.txt')
    df['id'] = df['StudyID']
    formula = "methylation ~ asthma + age + gender"

    np.random.seed(10)
    ntrue, nfalse = 10, 10

    results = []
    for fn in (bump_cluster, liptak_cluster, zscore_cluster):
        results.append(evaluate_method(clust_iter, df, formula, 'asthma', fn,
            ntrue, nfalse))

    formula = "methylation ~ asthma + age + gender"
    for fn in (gee_cluster, mixed_model_cluster):
        results.append(evaluate_method(clust_iter, df, formula, 'asthma', fn,
            ntrue, nfalse))

    results = pd.DataFrame(results)
    print pd.melt(results,
            id_vars=[c for c in results.columns if not ('false' in c or 'true' in c)],
            value_vars=[c for c in results.columns if 'false' in c or 'true' in
                c], value_name='n_lt_alpha')


