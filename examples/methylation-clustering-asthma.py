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
from itertools import starmap

import pandas as pd
import numpy as np
import scipy.stats as ss

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
    idx = [i for i, par in enumerate(res.model.exog_names)
                       if par.startswith(coef)]
    return {'p': res.pvalues[idx[0]],
            't': res.tvalues[idx[0]],
            'coef': res.params[idx[0]]}

def gee_cluster(formula, methylation, covs, coef, cov_struct=Exchangeable(),
        family=Gaussian()):
    cov_rep = pd.concat((covs for i in range(len(methylation))))
    nr, nc = methylation.shape
    cov_rep['CpG'] = np.repeat(['CpG_%i' % i for i in range(methylation.shape[0])],
                        methylation.shape[1])
    cov_rep['methylation'] = np.concatenate(methylation)

    res = GEE.from_formula(formula, groups=cov_rep['id'], data=cov_rep, cov_struct=cov_struct,
            family=family).fit()
    idx = [i for i, par in enumerate(res.model.exog_names)
                       if par.startswith(coef)]
    return {'p': res.pvalues[idx[0]],
            't': res.tvalues[idx[0]],
            'coef': res.params[idx[0]]}

def mixed_model_cluster(formula, methylation, covs, coef):
    """TODO."""
    covs['id'] = ['id_%i' % i for i in range(len(covs))]
    cov_rep = pd.concat((covs for i in range(len(methylation))))
    nr, nc = methylation.shape
    cov_rep['CpG'] = np.repeat(['CpG_%i' % i for i in range(methylation.shape[0])],
                        methylation.shape[1])
    cov_rep['methylation'] = np.concatenate(methylation)
    #cov2 = cov_rep[['id', 'CpG', 'methylation', 'age', 'gender', 'asthma']].to_csv('/tmp/m.csv', index=False)

    res = MixedLM.from_formula(formula, groups='id', re_formula="id + CpG", data=cov_rep)

    #res.set_random(cov_rep['CpG']) # TODO make them independent
    res = res.fit(free=(np.ones(res.k_fe), np.eye(res.k_re)))
    idx = [i for i, par in enumerate(res.model.exog_names)
                       if par.startswith(coef)]
    return {'p': res.pvalues[idx[0]],
            't': res.tvalues[idx[0]],
            'coef': res.params[idx[0]]}

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
    return dict(t=np.mean([r.tvalues[idx] for r in res]),
                coef=np.mean([r.params[idx] for r in res]),
                p=pvals, 
                corr=np.abs(ss.spearmanr(methylations.T)[0]))

def liptak_cluster(formula, methylations, covs, coef, family=Gaussian()):
    r = _combine_cluster(formula, methylations, covs, coef, family=family)
    r['p'] = stouffer_liptak(r['p'], r['corr'])
    return r

def zscore_cluster(formula, methylations, covs, coef, family=Gaussian()):
    r = _combine_cluster(formula, methylations, covs, coef, family=family)
    z, L = np.mean(norm.isf(r['p'])), len(r['p'])
    sz = 1.0 / L * np.sqrt(L + 2 * np.tril(r['corr'], k=-1).sum())
    r['p'] = norm.sf(z/sz)
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

def model_clusters(clust_iter, clin_df, model_str, coef, model_fn=gee_cluster):
    for r in ts.pmap(wrapper, ((model_fn, model_str, cluster, clin_df, coef)
                    for cluster in clust_iter), n=3):
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

    
    clusters = model_clusters(clust_iter, df, formula, "asthma", model_fn=liptak_cluster)

    for i, c in enumerate(clusters):
        print fmt.format(**c)
        if i > 1000: break

    #from cruzdb import Genome
    #g = Genome('sqlite:///hg19.db')
    #g.annotate((x.split("\t") for x in clusters), ('refGene', 'cpgIslandExt'), feature_strand=True)

