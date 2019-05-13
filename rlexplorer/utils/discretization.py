import numpy as np
from numpy.lib.histograms import _get_outer_edges

# range is a keyword argument to many functions, so save the builtin so they can
# use it.
_range = range


def create_edges(bins, sample=None, range=None):
    D = len(bins)
    nbin = np.empty(D, int)
    edges = D * [None]

    if sample is None:
        sample = np.empty((1, D))

    # Create edge arrays
    for i in _range(D):
        if np.ndim(bins[i]) == 0:
            if bins[i] < 1:
                raise ValueError(
                    '`bins[{}]` must be positive, when an integer'.format(i))
            smin, smax = _get_outer_edges(sample[:, i], range[i])
            edges[i] = np.linspace(smin, smax, bins[i] + 1)
        elif np.ndim(bins[i]) == 1:
            edges[i] = np.asarray(bins[i])
            if np.any(edges[i][:-1] > edges[i][1:]):
                raise ValueError(
                    '`bins[{}]` must be monotonically increasing, when an array'
                        .format(i))
        else:
            raise ValueError(
                '`bins[{}]` must be a scalar or 1d array'.format(i))

        nbin[i] = len(edges[i]) + 1  # includes an outlier on each end

    return edges, nbin


def get_position2(sample, edges, nbin):
    D = len(edges)
    # Compute the bin number each sample falls into.
    Ncount = tuple(
        # avoid np.digitize to work around gh-11022
        np.searchsorted(edges[i], sample[:, i], side='right')
        for i in _range(D)
    )
    # Compute the sample indices in the flattened histogram matrix.
    # This raises an error if the array is too large.
    xy = np.ravel_multi_index(Ncount, nbin)
    return xy


class Discretizer:
    def __init__(self, bins, range=None):
        """
        Parameters
        ----------
        bins : sequence or int, optional
            The bin specification:

            * A sequence of arrays describing the bin edges along each dimension.
            * The number of bins for each dimension (nx, ny, ... =bins)
            * The number of bins for all dimensions (nx=ny=...=bins).

        range : sequence, optional
            A sequence of length D, each an optional (lower, upper) tuple giving
            the outer bin edges to be used if the edges are not given explicitly in
            `bins`.
            An entry of None in the sequence results in the minimum and maximum
            values being used for the corresponding dimension.
            The default, None, is equivalent to passing a tuple of D None values.
        """
        try:
            D = len(bins)
        except TypeError:
            D = 1

        # normalize the range argument
        if range is None:
            range = (None,) * D
        elif len(range) != D:
            raise ValueError('range argument must have one entry per dimension')

        self.edges, self.nbin = create_edges(bins, range=range)

    def dpos(self, sample):
        D = len(self.edges)
        sample = np.reshape(sample, (-1, D))

        return np.asscalar(get_position2(sample=sample, edges=self.edges, nbin=self.nbin))

    def n_bins(self):
        return np.prod(self.nbin)


class UniformDiscWOB(Discretizer):
    def __init__(self, bins, binstoremove, range=None):
        super(UniformDiscWOB, self).__init__(bins=bins, range=range)
        self.binstoremove = np.array(binstoremove)

    def dpos(self, sample):
        position = super(UniformDiscWOB, self).dpos(sample=sample)
        shift = np.sum(self.binstoremove <= position)
        return max(0, np.asscalar(position - shift))

    def n_bins(self):
        return np.prod(self.nbin) - len(self.binstoremove)

    def available(self, sample):
        position = super(UniformDiscWOB, self).dpos(sample=sample)
        return position not in self.binstoremove
