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
import pickle

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
    a = np.array(a)
    a_mat = array_rotate(a, shift, step)
    d = a_mat - np.array(b)
    d = np.sqrt((d*d).sum(axis=1))
    return d.min()


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


def saveOracle(oracle,fileName):
    with open(fileName, 'wb') as output:
        pickle.dump(oracle, output, pickle.HIGHEST_PROTOCOL)


def calculateDistance(a,b,c):
    return abs(a-b) + abs(b-c)


def calculatedistance(a,b):
    return abs(a-b)

# def findNonEmpty(sfx1 = [], sfx2 = [], sfx3 = []):
#     if not sfx2 and not sfx3 and sfx1:
#         return min(sfx1)

#     if not sfx1 and not sfx3 and sfx2:
#         return min(sfx2)

#     if not sfx2 and not sfx1 and sfx3:
#         return min(sfx3)


def findTriplets(sfx1, sfx2, sfx3):
    i,j,k = 0,0,0
    indices = [0,0,0]
    min_value = float('Inf')

    while(i < len(sfx1) and j < len(sfx2) and k < len(sfx3)):
        dist = calculateDistance(sfx1[i],sfx2[j],sfx3[k])

        if dist < min_value:
            min_value = dist
            indices[0],indices[1],indices[2] = i,j,k

        if sfx1[i]<sfx2[j] and sfx1[i]<sfx3[k] and i<len(sfx1):
            i += 1

        elif sfx2[j]<sfx3[k] and j<len(sfx2):
            j += 1

        else:
            k += 1
    
    return [sfx1[indices[0]],sfx2[indices[1]],sfx3[indices[2]]]


def findDoublets(sfx1, sfx2):
    i,j= 0,0
    indices = [0,0]
    min_value = float('Inf')

    while(i < len(sfx1) and j < len(sfx2)):
        dist = calculatedistance(sfx1[i],sfx2[j])

        if dist < min_value:
            min_value = dist
            indices[0],indices[1] = i,j

        if sfx1[i]<sfx2[j] and i<len(sfx1):
            i += 1

        else:
            j += 1
    
    return [sfx1[indices[0]],sfx2[indices[1]]]