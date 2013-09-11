Aclust
======
Streaming agglomerative clustering with custom distance and correlation


*Agglomerative clustering* is a very simple algorithm.
The function `aclust` provided here is an attempt at a simple implementation
of a modified version that allows a stream of input so that data is not
required to be read into memory all at once. Most clustering algorithms operate
on a matrix of correlations which may not be feasible with high-dimensional
data.

`aclust` **defers** some complexity to the caller by relying on a stream of
objects that support an interface (I know, I know) of:

    obj.distance(other) -> numeric
    obj.is_correlated(other) -> bool

While this does add some infrastructure, we can imagine a class with
position and values attributes, where the former is an integer and the
latter is a list of numeric values. Then, those methods would be implemented
as:

    def distance(self, other):
        return self.position - other.position

    def is_correlated(self, other):
        return np.corrcoef(self.values, other.values)[0, 1] > 0.5

This allows the `aclust` function to be used on **any** kind of data. We can
imagine that distance might return the Levenshtein distance between 2 strings
while is\_correlated might indicate their presence in the same sentence or in
sentences with the same sentiment.

Since the input can be- and the output is- streamed, it is assumed the the objs
are in sorted order. This is important for things like genomic data, but may be
less so in text, where the max\_skip parameter can be set to a large value to
determine how much data is kept in memory.

See the function docstring for examples and options. The function signature is:

   aclust(object\_stream, max\_dist, min\_clust\_size=0,
          max\_skip=1, corr\_with=any)

It yields clusters (lists) of objects from the input object stream.

Uses
====

+  Clustering methylation data which we know to be locally correlated. We can
   use this to reduce the number of tests (of association) from 1 test per CpG,
   to 1 test per correlated unit.
   See: https://github.com/brentp/aclust/blob/master/examples/methylation-clustering-asthma.py for a full example.

```
    chrom   start   end n_probes   probes                asthma.pvalue   asthma.tstat    asthma.coef
    chr1    566570  567501  8   chr1:566570,chr1:566731,chr1:567113,chr1:567206,chr1:567312,chr1:567348,chr1:567358,chr1:567501 0.4566  -0.74   -0.06
    chr1    713985  714021  3   chr1:713985,chr1:714012,chr1:714021 0.1185  -1.56   -0.13
    chr1    845810  846195  3   chr1:845810,chr1:846155,chr1:846195 0.5913  0.54    0.04
    chr1    848379  848440  3   chr1:848379,chr1:848409,chr1:848440 0.3399  -0.95   -0.06
    chr1    854766  855046  7   chr1:854766,chr1:854824,chr1:854838,chr1:854918,chr1:854951,chr1:854966,chr1:855046 0.7482  -0.32   -0.02
    chr1    870791  871546  8   chr1:870791,chr1:870810,chr1:870958,chr1:871033,chr1:871057,chr1:871308,chr1:871441,chr1:871546 0.2198  -1.23   -0.11
    chr1    892857  892948  3   chr1:892857,chr1:892914,chr1:892948 0.2502  -1.15   -0.05
    chr1    901062  901799  5   chr1:901062,chr1:901449,chr1:901685,chr1:901725,chr1:901799 0.6004  0.52    0.04
    chr1    946875  947091  4   chr1:946875,chr1:947003,chr1:947018,chr1:947091 0.9949  0.01    0.00
```
   So we can filter on the asthma.pvalue to find regions associated with asthma.
  

INSTALL
=======

`aclust` is available on pypi, as such it can be installed with:

    pip install aclust


Acknowledgments
===============

The idea of this is taken from this paper:

    Sofer, T., Schifano, E. D., Hoppin, J. A., Hou, L., & Baccarelli, A. A. (2013). A-clustering: A Novel Method for the Detection of Co-regulated Methylation Regions, and Regions Associated with Exposure. Bioinformatics, btt498.

The example uses a pull-request implementing GEE for python's statsmodels:
    https://github.com/statsmodels/statsmodels/pull/928

