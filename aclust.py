"""
Streaming agglomerative clustering with custom distance and correlation
functions.
"""

__version__ = "0.1.2"

def _get_linkage_function(linkage):
    """
    >>> "f_linkage" in str(_get_linkage_function(0.5))
    True
    >>> "i_linkage" in str(_get_linkage_function(2))
    True
    >>> any is _get_linkage_function('single')
    True
    >>> all is _get_linkage_function('complete')
    True

    >>> ff = _get_linkage_function(0.5)
    >>> ff([True, False, False])
    False
    >>> ff([True, True, False])
    True

    >>> fi = _get_linkage_function(3)
    >>> fi([True, False, False])
    False
    >>> fi([True, True, False])
    False
    >>> fi([True, True, True]) and fi([True] * 10)
    True
    """

    if linkage == 'single':
        return any

    if linkage == 'complete':
        return all

    if isinstance(linkage, float):
        assert 0 < linkage <= 1
        if linkage == 1:
            return all

        def f_linkage(bools, p=linkage):
            v = list(bools)
            if len(v) == 0: return False
            return (sum(v) / float(len(v))) >= p
        return f_linkage

    if isinstance(linkage, int):
        assert linkage >= 1
        if linkage == 1:
            return any

        def i_linkage(bools, n=linkage):
            v = list(bools)
            return sum(v) >= min(len(v), n)
        return i_linkage
    1/0

def aclust(objs, max_dist, max_skip=1, linkage='single', multi_member=False):
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

    max_skip: 1 allows to skip one cluster which is not correlated with
              the current object and check the next (more distant) cluster.

    linkage: defines requirements for a new object to be added to an existing
             cluster. One of:
               'single': meaning an object is added to a cluster if it
                         correlated with any object in that cluster.
               'complete': meaning it must be correlated will all objects in
                           the cluster
               <integer>: it must be associated with at least this many objects
                          in the cluster (the min of this value and the number
                          of objects.)
               <float>: it must be associate with at least this portion of
                        objects in the cluster
    multi_member: whether a feature be a member of multiple clusters.
                  False: can only be a member of the nearest cluster to which
                         it has a correlation
                  True: can be a member of any cluster with which it is
                        correlated (and within a given distance)

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


    # test skipping
    >>> for c in aclust([Feature(-1, range(5)[::-1])] + feats + \
    ...                 [Feature( 2, range(5)[::-1])],
    ...                  max_dist=1, max_skip=4):
    ...     if len(c) > 1:
    ...         print c
    [(0, [0, 1, 2, 3, 4]), (1, [0, 1, 2, 3, 4])]

    # with maximum dist set high as well...
    >>> for c in aclust([Feature(-1, range(5)[::-1])] + feats + \
    ...                 [Feature( 2, range(5)[::-1])],
    ...                  max_dist=4, max_skip=4):
    ...     if len(c) > 1:
    ...         print c
    [(-1, [4, 3, 2, 1, 0]), (2, [4, 3, 2, 1, 0])]
    [(0, [0, 1, 2, 3, 4]), (1, [0, 1, 2, 3, 4])]
    """

    linkage = _get_linkage_function(linkage)

    objs = iter(objs)
    # accumulate clusters here.
    clusters = [[objs.next()]]

    for obj in objs:

        # clean out our list of clusters
        while len(clusters) > 0 and obj.distance(clusters[0][-1]) > max_dist:
            yield clusters.pop(0)

        while len(clusters) > max_skip:
            yield clusters.pop(0)

        # check against all clusters. closest first.
        any_cluster = False
        for clust in clusters[::-1]:
            inear = (i for i, r in enumerate(clust) if obj.distance(r) <=
                    max_dist)
            if linkage(obj.is_correlated(clust[i]) for i in inear):
                clust.append(obj)
                any_cluster = True
                if not multi_member: break
        if not any_cluster:
            # didn't get to any cluster. make a new one...
            clusters.append([obj])

    for clust in clusters:
        yield clust

def test():
    import doctest
    return doctest.testmod(__import__(__name__))

if __name__ == "__main__":
    import doctest
    print doctest.testmod()
