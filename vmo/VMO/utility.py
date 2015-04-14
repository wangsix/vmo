'''
utility.py
Variable Markov Oracle in python

@copyright: 
Copyright (C) 3.2015 Cheng-i Wang

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
@contact: wangsix@gmail.com, chw160@ucsd.edu
'''
import numpy as np

def entropy(x):
    x = np.divide(x, sum(x), dtype = float)
    return sum(np.multiply(-np.log2(x),x))

def array_rotate(a):
    _a = a
    for _i in range(1,a.size):
        _a = np.vstack((_a, np.roll(a,_i)))
    return _a
        
def trnspose_inv(a, b_vec):
    d_vec = []
    a = np.array(a)
    a_mat = array_rotate(a)
    for b in b_vec:
        d = a_mat - np.array(b)
        d = np.sqrt((d*d).sum(axis=1))
        d_vec.append(d.min())
    return np.array(d_vec)
