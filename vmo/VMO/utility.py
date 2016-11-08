"""
utility.py
Variable Markov Oracle in python

@copyright: 
Copyright (C) 7.2015 Cheng-i Wang,  Theis Bazin

This file is part of vmo.

@license: 
vmo is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

vmo is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with vmo.  If not, see <http://www.gnu.org/licenses/>.
@author: Cheng-i Wang
@contact: wangsix@gmail.com, chw160@ucsd.edu, tbazin@eng.ucsd.edu
"""

import numpy as np
import scipy.stats as stats
import sklearn.preprocessing as pre
import scipy.spatial.distance as dist


def entropy(x):
    return stats.entropy(x)


def array_rotate(a, shift=1, step=None):
    _a = a
    if step is None and step <= 1:
        for _i in range(1, a.size):
            _a = np.vstack((_a, np.roll(a, _i)))
    else:
        up_array = [x * shift for x in range(1, step + 1)]
        down_array = [-x * shift for x in range(1, step + 1)]
        up_array.extend(down_array)
        for _i in up_array:
            _a = np.vstack((_a, np.roll(a, _i)))
    return _a


def transpose_inv(a, b_vec, shift=1, step=None):
    d_vec = []
    a = np.array(a)
    a_mat = array_rotate(a, shift, step)
    for b in b_vec:
        d = a_mat - np.array(b)
        d = np.sqrt((d * d).sum(axis=1))
        d_vec.append(d.min())
    return np.array(d_vec)


r_fifth = 1.
r_minor_thirds = 1.
r_major_thirds = 0.5


def _make_tonnetz_matrix():
    """Return the tonnetz projection matrix."""
    pi = np.pi
    chroma = np.arange(12)

    # Define each row of the transform matrix
    fifth_x = r_fifth * (np.sin((7 * pi / 6) * chroma))
    fifth_y = r_fifth * (np.cos((7 * pi / 6) * chroma))
    minor_third_x = r_minor_thirds * (np.sin(3 * pi / 2 * chroma))
    minor_third_y = r_minor_thirds * (np.cos(3 * pi / 2 * chroma))
    major_third_x = r_major_thirds * (np.sin(2 * pi / 3 * chroma))
    major_third_y = r_major_thirds * (np.cos(2 * pi / 3 * chroma))

    # Return the tonnetz matrix
    return np.vstack((fifth_x, fifth_y,
                      minor_third_x, minor_third_y,
                      major_third_x, major_third_y))


# Define a global constant to avoid recomputations of the Tonnetz matrix
__TONNETZ_MATRIX = _make_tonnetz_matrix()


def _to_tonnetz(chromagram):
    """Project a chromagram on the tonnetz.
    Returned value is normalized to prevent numerical instabilities.
    """
    if np.sum(np.abs(chromagram)) == 0.:
        # The input is an empty chord, return zero.
        return np.zeros(6)

    _tonnetz = np.dot(chromagram, __TONNETZ_MATRIX.T)
    # print _tonnetz.shape
    # one_norm = np.sum(np.abs(_tonnetz))  # Non-zero value
    # _tonnetz = _tonnetz / float(one_norm)  # Normalize tonnetz vector
    # _tonnetz = pre.normalize(_tonnetz, axis=1)
    return _tonnetz


def tonnetz_dist(a, b):
    """Compute tonnetz-distance between two chromagrams.

    ----
    >>> C = np.zeros(12)
    >>> C[0] = 1
    >>> D = np.zeros(12)
    >>> D[2] = 1
    >>> G = np.zeros(12)
    >>> G[7] = 1
    The distance is zero on equivalent chords
    >>> tonnetz_dist(C, C) == 0
    True
    The distance is symetric
    >>> tonnetz_dist(C, D) == tonnetz_dist(D, C)
    True
    >>> tonnetz_dist(C, D) > 0
    True
    >>> tonnetz_dist(C, G) < tonnetz_dist(C, D)
    True
    """
    # [a_tonnetz, b_tonnetz] = [_to_tonnetz(x) for x in [a.reshape((1,-1)), b]]
    [a_tonnetz, b_tonnetz] = [_to_tonnetz(x) for x in [a, b]]
    return dist.cdist(a_tonnetz, b_tonnetz, metric='cosine')
    # return np.linalg.norm(b_tonnetz - a_tonnetz)


def get_sfx(oracle, s_set, k):
    while oracle.sfx[k] != 0:
        s_set.add(oracle.sfx[k])
        k = oracle.sfx[k]
    return s_set


def get_rsfx(oracle, rs_set, k):
    if not oracle.rsfx[k]:
        return rs_set
    else:
        rs_set = rs_set.union(oracle.rsfx[k])
        for _k in oracle.rsfx[k]:
            rs_set = rs_set.union(get_rsfx(oracle, rs_set, _k))
        return rs_set
