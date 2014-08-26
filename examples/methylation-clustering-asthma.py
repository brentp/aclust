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
from aclust import aclust
import toolshed as ts
from itertools import starmap, izip

import pandas as pd
import numpy as np
import scipy.stats as ss
import patsy

from scipy.stats import norm
from numpy.linalg import cholesky as chol

from statsmodels.api import GEE, GLM, MixedLM
from statsmodels.genmod.dependence_structures import Exchangeable
from statsmodels.genmod.families import Gaussian

def one_cluster(formula, methylation, covs, coef, family=Gaussian()):
    """used when we have a "cluster" with 1 probe."""
    c = covs.copy()
    c['methylation'] = methylation
    res = GLM.from_formula(formula, data=c, family=family).fit()
    return get_ptc(res, coef)

def gee_cluster(formula, methylation, covs, coef, cov_struct=Exchangeable(),
        family=Gaussian()):
    cov_rep = pd.concat((covs for i in range(len(methylation))))
    nr, nc = methylation.shape
    cov_rep['CpG'] = np.repeat(['CpG_%i' % i for i in range(methylation.shape[0])],
                        methylation.shape[1])
    cov_rep['methylation'] = np.concatenate(methylation)

    res = GEE.from_formula(formula, groups=cov_rep['id'], data=cov_rep, cov_struct=cov_struct,
            family=family).fit()
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

def _combine_cluster(formula, methylations, covs, coef, family=Gaussian()):
    """function called by z-score and liptak to get pvalues"""
    res = [GLM.from_formula(formula, covs, family=family).fit()
        for methylation in methylations]

    idx = [i for i, par in enumerate(res[0].model.exog_names)
                   if par.startswith(coef)][0]
    pvals = np.array([r.pvalues[idx] for r in res], dtype=np.float64)
    pvals[pvals == 1] = 1.0 - 9e-16
    return dict(t=np.array([r.tvalues[idx] for r in res]),
                coef=np.array([r.params[idx] for r in res]),
                p=pvals,
                corr=np.abs(ss.spearmanr(methylations.T)[0]))

def liptak_cluster(formula, methylations, covs, coef, family=Gaussian()):
    r = _combine_cluster(formula, methylations, covs, coef, family=family)
    r['p'] = stouffer_liptak(r['p'], r['corr'])
    r['t'], r['coef'] = r['t'].mean(), r['coef'].mean()
    return r

def zscore_cluster(formula, methylations, covs, coef, family=Gaussian()):
    r = _combine_cluster(formula, methylations, covs, coef, family=family)
    z, L = np.mean(norm.isf(r['p'])), len(r['p'])
    sz = 1.0 / L * np.sqrt(L + 2 * np.tril(r['corr'], k=-1).sum())
    r['p'] = norm.sf(z/sz)
    r['t'], r['coef'] = r['t'].mean(), r['coef'].mean()
    return r

def wrapper(model_fn, model_str, cluster, clin_df, coef):
    """wrap the user-defined functions to return everything we expect and
    to call just GLM when there is a single probe."""
    if len(cluster) > 1:
        r = model_fn(model_str, np.array([c.values for c in cluster]), clin_df, coef)
    else:
        r = one_cluster(model_str, cluster[0].values, clin_df, coef)
    r['chrom'] = cluster[0].chrom
    r['start'] = cluster[0].position
    r['end'] = cluster[-1].position
    r['n_probes'] = len(cluster)
    r['probes'] = ",".join(c.spos for c in cluster)
    r['var'] = coef
    return r

def get_coef(c, cutoff):
    if c < 0: return min(0, c + cutoff)
    return max(0, c - cutoff)

def bump_cluster(model_str, methylations, covs, coef, cutoff=0.025, nsims=1000):
    orig = _combine_cluster(model_str, methylations, covs, coef)
    obs_coef = sum(get_coef(c, cutoff) for c in orig['coef'])

    reduced_residuals, reduced_fitted = [], []

    # get the reduced residuals and models so we can shuffle
    for i, methylation in enumerate(methylations):
        y, X = patsy.dmatrices(model_str, covs, return_type='dataframe')
        idxs = [par for par in X.columns if par.startswith(coef)]
        assert len(idxs) == 1, ('too many coefficents like', coef)
        X.pop(idxs[0])
        fitr = GLM(y, X).fit()

        reduced_residuals.append(np.array(fitr.resid_response))
        reduced_fitted.append(np.array(fitr.fittedvalues))

    ngt, idxs = 0, np.arange(len(methylations[0]))

    for isim in range(nsims):
        np.random.shuffle(idxs)

        fakem = np.array([rf + rr[idxs] for rf, rr in izip(reduced_fitted,
            reduced_residuals)])
        assert fakem.shape == methylations.shape

        sim = _combine_cluster(model_str, fakem, covs, coef)
        ccut = sum(get_coef(c, cutoff) for c in sim['coef'])
        ngt += abs(ccut) > abs(obs_coef)
        # progressive monte-carlo.
        if ngt > 5: break

    p = (1.0 + ngt) / (2.0 + isim)
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
        self.chrom, self.position, self.values = chrom, pos, values
        self.spos = "%s:%i" % (chrom, pos)

    def distance(self, other):
        if self.chrom != other.chrom: return sys.maxint
        return self.position - other.position

    def is_correlated(self, other):
        rho, p = ss.spearmanr(self.values, other.values)
        return rho > 0.7

    def __repr__(self):
        return str((self.position, self.values))

    def __cmp__(self, other):
        return cmp(self.chrom, other.chrom) or cmp(self.position,
                                                   other.position)

if __name__ == "__main__":

    def feature_gen(fname):
        for i, toks in enumerate(ts.reader(fname, header=False)):
            if i == 0: continue
            chrom, pos = toks[0].split(":")
            yield Feature(chrom, int(pos), map(float, toks[1:]))

    fmt = "{chrom}\t{start}\t{end}\t{n_probes}\t{p:5g}\t{t:.4f}\t{coef:.4f}\t{probes}\t{var}"
    print ts.fmt2header(fmt)

    clust_iter = (c for c in aclust(feature_gen(sys.argv[1]),
                                    max_dist=400, max_skip=2) if len(c) > 2)


    df = pd.read_table('meth.clin.txt')
    df['id'] = df['StudyID']

    formula = "methylation ~ asthma + age + gender"

    clusters = model_clusters(clust_iter, df, formula, "asthma",
            model_fn=zscore_cluster)

    for i, c in enumerate(clusters):
        print fmt.format(**c)
        if i > 10: break

    #from cruzdb import Genome
    #g = Genome('sqlite:///hg19.db')
    #g.annotate((x.split("\t") for x in clusters), ('refGene', 'cpgIslandExt'), feature_strand=True)

