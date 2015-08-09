"""
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
"""

import numpy as np
import scipy
import librosa
import sklearn.cluster
import sklearn.mixture
import scipy.stats as stats
import scipy.cluster.hierarchy as scihc
import editdistance as edit


def entropy(x):
    return scipy.stats.entropy(x, base=2)


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


def normalized_graph_laplacian(mat):
    mat_inv = 1. / np.sum(mat, axis=1)
    mat_inv[~np.isfinite(mat_inv)] = 1.
    mat_inv = np.diag(mat_inv ** 0.5)
    laplacian = np.eye(len(mat)) - mat_inv.dot(mat.dot(mat_inv))

    return laplacian


def eigen_decomposition(mat, k=11):  # Changed from 11 to 8 then to 6(7/22)
    vals, vecs = scipy.linalg.eig(mat)
    vals = vals.real
    vecs = vecs.real
    idx = np.argsort(vals)

    vals = vals[idx]
    vecs = vecs[:, idx]

    if len(vals) < k + 1:
        k = -1

    return vecs[:, :k].T


def edit_distance(u, v):
    return float(edit.eval(u, v))


def normalized_edit_distance(u, v):
    return edit_distance(u, v)/np.max([len(u), len(v)])


"""Adapted from Brian McFee`s spectral clustering algorithm for music structural segmentation
https://github.com/bmcfee/laplacian_segmentation
"""



def segment_labeling(x, boundaries, k=0.05):

    x_sync = librosa.feature.sync(x.T, boundaries)
    # d = dist.pdist(x_sync)
    z = scihc.linkage(x_sync.T, method='ward')
    t = k * np.max(z[:, 2])
    seg_labels = scihc.fcluster(z, t=t, criterion='distance')

    # c = sklearn.cluster.KMeans(n_clusters=k, tol=1e-8)
    # seg_labels = c.fit_predict(x_sync.T)

    return seg_labels


def find_boundaries(frame_labels, width=9):
    frame_labels = np.pad(frame_labels, (width/2, width/2+1), mode='reflect')
    frame_labels = np.array([stats.mode(frame_labels[i:j])[0][0]
                             for (i, j) in zip(range(0, len(frame_labels)-width),
                                               range(width, len(frame_labels)))])
    boundaries = 1 + np.asarray(np.where(frame_labels[:-1] != frame_labels[1:])).reshape((-1,))
    boundaries = np.unique(np.concatenate([[0], boundaries, [len(frame_labels)]]))
    return boundaries


def boundaries_adjustment(oracle, boundaries, labels):

    _tmp_boundary = np.insert(boundaries, 0, -8.0)
    b_distance = np.diff(_tmp_boundary)
    boundaries = boundaries[b_distance > 4]
    labels = labels[np.diff(boundaries) > 4]

    feature = oracle.f_array[1:]
    new_boundaries = [boundaries[0]]
    for b in boundaries[1:-1]:
        if b < 8:
            neighbor_feature = feature[:b+5]
            adj = -(b-1)
        elif len(feature)-b < 8:
            neighbor_feature = feature[b-8:]
            adj = -7
        else:
            neighbor_feature = feature[b-8:b+5]
            adj = -7
        offset = np.argmax(np.sum(np.square(np.diff(neighbor_feature, axis=0)), axis=1)) + adj
        new_b = b + offset
        new_boundaries.append(new_b)
    new_boundaries.append(boundaries[-1])

    new_boundaries = np.array(new_boundaries)
    _tmp_boundary = np.insert(new_boundaries, 0, -8.0)
    print _tmp_boundary
    b_distance = np.diff(_tmp_boundary)
    print b_distance
    print new_boundaries
    new_boundaries = new_boundaries[b_distance > 4]
    labels = labels[np.diff(new_boundaries) > 4]

    return new_boundaries, labels
    # return boundaries, labels