"""
Streaming agglomerative clustering with custom distance and correlation
functions.
"""

def aclust(objs, max_dist, min_clust_size=0, max_skip=1, corr_with=any):
    r"""
    objs: must be sorted and could (should) be a lazy iterable.
          each obj in objs must have this interface (I know, I know):
          + obj.distance(other_obj) which returns an integer
          + obj.is_correlated(other_obj) which returns a bool

          groups of objs, such as those from distinct chromosomes from genomic
          data should be separated before sending to this function. (or the
          distance function could return a value > max_dist when chromosomes
          are not equal

    max_dist: maximum distance at which to cluster 2 objs (assuming they
              are correlated.

    min_clust_size: only yield clusters of at least this size.

    max_skip: 1 allows to skip one cluster which is not correlated with
              the current object and check the next (more distant) cluster.

    min_clust_size: only yield clusters that have at least this many
                    members.

    corr_with: likely either `any` or `all`, but can be any function that
               takes an iterable of booleans and returns a boolean indicating
               whether an obj should be added to a cluster. `any` would mean
               that it must only be correlated with one of them (single-
               linkage) `all` would mean that it must be correlated with all of
               them (complete-linkage)

    Examples:
    First, the class that implements o.distance(other) and
    o.is_correlated(other)

    >>> import numpy as np
    >>> class Feature(object):
    ...     def __init__(self, pos, values):
    ...         self.position, self.values = pos, values
    ...     def distance(self, other):
    ...         return self.position - other.position
    ... 
    ...     def is_correlated(self, other):
    ...         return np.corrcoef(self.values, other.values)[0, 1] > 0.5
    ... 
    ...     def __repr__(self):
    ...         return str((self.position, self.values))

    # create 3 features of distance 1 apart all with same values
    >>> feats = [Feature(i, range(5)) for i in range(3)]

    # they all cluster together into a single cluster.
    >>> list(aclust(feats, max_dist=1))
    [[(0, [0, 1, 2, 3, 4]), (1, [0, 1, 2, 3, 4]), (2, [0, 1, 2, 3, 4])]]

    # unless the max_dist=0
    >>> list(aclust(feats, max_dist=0))
    [[(0, [0, 1, 2, 3, 4])], [(1, [0, 1, 2, 3, 4])], [(2, [0, 1, 2, 3, 4])]]

    >>> _ = feats.pop()

    # add a feature that's far from the others
    >>> list(aclust(feats + [Feature(8, range(5))], max_dist=1))
    [[(0, [0, 1, 2, 3, 4]), (1, [0, 1, 2, 3, 4])], [(8, [0, 1, 2, 3, 4])]]

    # add a feature that's not correlated with the others.
    >>> for c in (aclust(feats + [Feature(2, range(5)[::-1])], max_dist=1)):
    ...     print c
    [(0, [0, 1, 2, 3, 4]), (1, [0, 1, 2, 3, 4])]
    [(2, [4, 3, 2, 1, 0])]


    # only show bigger clusters
    >>> for c in aclust(feats + [Feature(2, range(5)[::-1])], max_dist=1,
    ...    min_clust_size=2):
    ...    print c
    [(0, [0, 1, 2, 3, 4]), (1, [0, 1, 2, 3, 4])]


    # test skipping
    >>> for c in aclust([Feature(-1, range(5)[::-1])] + feats + \
    ...                 [Feature( 2, range(5)[::-1])],
    ...                  max_dist=1, min_clust_size=2, max_skip=4):
    ...     print c
    [(0, [0, 1, 2, 3, 4]), (1, [0, 1, 2, 3, 4])]

    # with maximum dist set high as well...
    >>> for c in aclust([Feature(-1, range(5)[::-1])] + feats + \
    ...                 [Feature( 2, range(5)[::-1])],
    ...                  max_dist=4, min_clust_size=2, max_skip=4):
    ...     print c
    [(-1, [4, 3, 2, 1, 0]), (2, [4, 3, 2, 1, 0])]
    [(0, [0, 1, 2, 3, 4]), (1, [0, 1, 2, 3, 4])]
    """

    objs = iter(objs)
    last_obj = objs.next()
    # accumulate clusters here.
    clusters = [[last_obj]]
    for obj in objs:
        #assert obj.distance(last_obj) >= 0,
        #        ("input must be sorted by "position")

        # clean out our list of clusters
        while len(clusters) > 0 and obj.distance(clusters[0][-1]) > max_dist:
            c = clusters.pop(0)
            if len(c) >= min_clust_size:
                yield c

        while len(clusters) > max_skip:
            c = clusters.pop(0)
            if len(c) >= min_clust_size:
                yield c

        # check against all clusters. closest first.
        for clust in clusters[::-1]:
            inear = (i for i, r in enumerate(clust) if obj.distance(r) <=
                    max_dist)
            if corr_with(obj.is_correlated(clust[i]) for i in inear):
                clust.append(obj)
                break
        else:
            # didn't get to any cluster. make a new one...
            clusters.append([obj])

    for clust in (c for c in clusters if len(c) >= min_clust_size):
        yield clust


if __name__ == "__main__":
    import doctest
    doctest.testmod()

