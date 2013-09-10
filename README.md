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

Uses
====

+  Clustering methylation data which we know to be locally correlated. We can
   use this to reduce the number of tests (of association) from 1 test per CpG,
   to 1 test per correlated unit.


