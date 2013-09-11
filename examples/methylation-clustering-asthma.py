"""
an example of how to use the output from aclust.

For each cluster, we get N probes. Since those will be correlated within
an individuals, we use GEE's from a pull request on statsmodels.

We could also use, for example, a mixed-effect model, e.g., in lme4 syntax:

    methylation ~ asthma + age + (1|sample_id)

to allow a random intercept for each sample (where each sample will have
a number of measurements equal to the number of probes in a given cluster.
"""

import sys
from aclust import aclust
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.dependence_structures.varstruct import (Exchangeable,
        Independence, Autoregressive)
from statsmodels.genmod.families import Gaussian, Binomial
import pandas as pd


def ilogit(a):
    return 1 / (1 + np.exp(-a))

def gee_cluster(clust_iter, clin_df, model_str, coef, id_col,
        varstruct=Exchangeable(), family=Gaussian()):
    """
    clust_iter: an iterable as returned from aclust.aclust()
    clin_df: a pandas dataframe with the clinical data to be used in the model.
             rows from clin_df should match columns in the data sent to
             clust_iter.
    """

    print "#chrom\tstart\tend\tn_probes\tprobes\t{coef}.pvalue\t{coef}.tstat\t{coef}.coef"\
            .format(**locals())


    for cluster in clust_iter:

        assert len(cluster[0].values)

        methylation = np.concatenate([c.values for c in cluster])
        locs = np.concatenate([[c.position] * len(c.values) for c in cluster])
        df_rep = pd.concat((clin_df for i in range(len(cluster))))
        """
        # TODO: set this if autoregressive
        locs = np.zeros_like(methylation)
        for group in df.StudyID.unique():
            jj = np.flatnonzero(df_rep.StudyID == group)
            locs[jj] = range(len(jj))
        """
        res = GEE.from_formula(model_str, df_rep,
                               family=family,
                               time=locs,
                               groups=list(getattr(df_rep, id_col)),
                               varstruct=varstruct).fit()

        idx = [i for i, par in enumerate(res.model.exog_names)
                       if par.startswith(coef)]
        assert len(idx) == 1, (
            "should have a single coefficent matching %s" % coef,
            res.model.exog_names)

        print "%s\t%i\t%i\t%i\t%s\t%.4g\t%.2f\t%.2f" % (
            cluster[0].chrom,
            cluster[0].position,
            cluster[-1].position,
            len(cluster),
            ",".join(c.spos for c in cluster),
            res.pvalues[idx[0]],
            res.tvalues[idx[0]],
            res.params[idx[0]])


if __name__ == "__main__":

    from toolshed import reader
    import scipy.stats as ss
    import numpy as np

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
            return rho > 0.6

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

    df = pd.read_table('/home/brentp/src/denver-bio/2013/icac-nasal-epithelium/data/meth.clin.txt')
    formula = "methylation ~ asthma + age + gender + race_white + race_hispanic + race_aa"
    clust_iter = aclust(sorted(feature_gen()), max_dist=400, max_skip=2,
                        min_clust_size=3)

    gee_cluster(clust_iter, df, formula, "asthma", "StudyID",
                varstruct=Exchangeable())

