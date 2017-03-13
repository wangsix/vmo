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


def trnspose_inv(a, b, shift=1, step=None):
    d_vec = []
    a = np.array(a)
    a_mat = array_rotate(a, shift, step)
    d = a_mat - np.array(b)
    d = np.sqrt((d*d).sum(axis=1))
    return d.min()


# def transpose_inv(a, b_vec, shift=1, step=None):
#     d_vec = []
#     a = np.array(a)
#     a_mat = array_rotate(a, shift, step)
#     for b in b_vec:
#         d = a_mat - np.array(b)
#         d = np.sqrt((d * d).sum(axis=1))
#         d_vec.append(d.min())
#     return np.array(d_vec)


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
