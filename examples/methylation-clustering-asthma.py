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
from statsmodels.api import GEE, GLM
from statsmodels.genmod.dependence_structures import Exchangeable
from statsmodels.genmod.families import Gaussian
from toolshed import pmap
import pandas as pd

from scipy.stats import norm
from numpy.linalg import cholesky as chol
qnorm = norm.ppf
pnorm = norm.cdf


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

def stouffer_liptak(pvals, sigma):
    pvals = np.array(pvals, dtype=np.float64)
    pvals[pvals == 1] = 1.0 - 9e-16
    qvals = norm.isf(pvals).reshape(len(pvals), 1)
    C = np.asmatrix(chol(sigma)).I
    qvals = C * qvals
    Cp = qvals.sum() / np.sqrt(len(qvals))
    return norm.sf(Cp)

def liptak_cluster(formula, methylations, covs, coef, family=Gaussian()):

    res = [GLM.from_formula(formula, covs, family=family).fit()
            for methylation in methylations]

    idx = [i for i, par in enumerate(res[0].model.exog_names)
                       if par.startswith(coef)][0]
    pvals = [r.pvalues[idx] for r in res]
    return dict(t=np.mean([r.tvalues[idx] for r in res]),
                coef=np.mean([r.params[idx] for r in res]),
                p=stouffer_liptak(pvals, np.corrcoef(methylations)))

def wrapper(model_fn, model_str, cluster, clin_df, coef):
    r = model_fn(model_str, np.array([c.values for c in cluster]), clin_df, coef)
    r['chrom'] = cluster[0].chrom
    r['start'] = cluster[0].position
    r['end'] = cluster[-1].position
    r['n_probes'] = len(cluster)
    r['probes'] = ",".join(c.spos for c in cluster)
    return r

def model_clusters(clust_iter, clin_df, model_str, coef, model_fn=gee_cluster):

    yield "#chrom\tstart\tend\tn_probes\t{coef}.pvalue\t{coef}.tstat\t{coef}.coef\tprobes"\
            .format(**locals())


    for r in pmap(wrapper, ((model_fn, model_str, cluster, clin_df, coef) for cluster in clust_iter)):
    #for cluster in clust_iter:

    #    assert len(cluster[0].values)

    #    methylation = np.array([c.values for c in cluster])
    #    r = model_fn(model_str, methylation, clin_df, coef)
        yield r
        continue
        yield "%s\t%i\t%i\t%i\t%.4g\t%.2f\t%.2f\t%s" % (
            cluster[0].chrom,
            cluster[0].position,
            cluster[-1].position,
            len(cluster),
            r['p'], r['t'], r['coef'],
            ",".join(c.spos for c in cluster))

if __name__ == "__main__":

    from toolshed import reader
    import scipy.stats as ss
    import numpy as np

    # convert M-values to Beta's then use logistic regression.
    def ilogit(a, base=2):
        return 1 / (1 + base**-a)

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
            return str((self.position, self.values))

        def __cmp__(self, other):
            return cmp(self.chrom, other.chrom) or cmp(self.position,
                                                       other.position)
    def feature_gen():
        for i, toks in enumerate(reader(1, header=False)):
            if i == 0: continue
            chrom, pos = toks[0].split(":")
            yield Feature(chrom, int(pos), map(float, toks[1:]))

    df = pd.read_table('meth.clin.txt')
    df['id'] = df['StudyID']
    formula = "methylation ~ asthma + age + gender + race_white + race_hispanic + race_aa"
    clust_iter = (c for c in
                    aclust(feature_gen(), max_dist=400, max_skip=2) if
                    len(c) > 1)

    clusters = model_clusters(clust_iter, df, formula, "asthma", model_fn=gee_cluster)
    from cruzdb import Genome
    for i, c in enumerate(clusters):
        print c
        if i > 200: break

    #g = Genome('sqlite:///hg19.db')
    #g.annotate((x.split("\t") for x in clusters), ('refGene', 'cpgIslandExt'), feature_strand=True)

