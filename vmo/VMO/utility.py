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
# import scipy.spatial.distance as dist
import scipy.stats as stats
import scipy.cluster.hierarchy as scihc


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


def eigen_decomposition(mat, k=11):
    vals, vecs = scipy.linalg.eig(mat)
    vals = vals.real
    vecs = vecs.real
    idx = np.argsort(vals)

    vals = vals[idx]
    vecs = vecs[:, idx]

    if len(vals) < k + 1:
        k = -1

    return vecs[:, :k].T

"""Adopted from Brian McFee`s spectral clustering for structural segmentation"""


def clustering_by_entropy(eigen_vecs, k_min, width=33):
    best_score = -np.inf
    best_boundaries = [0, eigen_vecs.shape[1]]
    best_n_types = 1
    y_best = eigen_vecs[:1].T

    label_dict = {1: np.zeros(eigen_vecs.shape[1])}  # The trivial solution

    for n_types in range(2, 1+len(eigen_vecs)):
        y = librosa.util.normalize(eigen_vecs[:n_types].T, norm=2, axis=1)

        # Try to label the data with n_types
        c = sklearn.cluster.KMeans(n_clusters=n_types, n_init=100)
        labels = c.fit_predict(y)
        label_dict[n_types] = labels

        # Find the label change-points
        boundaries = find_boundaries(labels, width)

        # boundaries now include start and end markers; n-1 is the number of segments
        feasible = (len(boundaries) > k_min)

        values = np.unique(labels)
        hits = np.zeros(len(values))

        for v in values:
            hits[v] = np.sum(values == v)

        hits = hits / hits.sum()

        score = entropy(hits) / np.log2(n_types)

        if score > best_score and feasible:
            best_boundaries = boundaries
            best_n_types = n_types
            best_score = score
            y_best = y


    # Did we fail to find anything with enough boundaries?
    # Take the last one then
    if best_boundaries is None:
        best_boundaries = boundaries
        best_n_types = n_types
        y_best = librosa.util.normalize(eigen_vecs[:best_n_types].T, norm=2, axis=1)

    # Classify each segment centroid

    labels = segment_labeling(y_best, best_boundaries, best_n_types)

    # intervals = zip(boundaries[:-1], boundaries[1:])
    best_labels = labels

    return best_boundaries, best_labels


def segment_labeling(x, boundaries, k):

    x_sync = librosa.feature.sync(x.T, boundaries)
    # d = dist.pdist(x_sync)
    z = scihc.linkage(x_sync.T, method='ward')
    t = 0.05 * np.max(z[:, 2])
    seg_labels = scihc.fcluster(z, t=t, criterion='distance')

    # c = sklearn.cluster.KMeans(n_clusters=k, tol=1e-8)
    # seg_labels = c.fit_predict(x_sync.T)

    return seg_labels


def find_boundaries(frame_labels, width=33):
    frame_labels = np.pad(frame_labels, (width/2, width/2+1), mode='reflect')
    frame_labels = np.array([stats.mode(frame_labels[i:j])[0][0]
                             for (i, j) in zip(range(0, len(frame_labels)-width),
                                               range(width, len(frame_labels)))])
    boundaries = 1 + np.asarray(np.where(frame_labels[:-1] != frame_labels[1:])).reshape((-1,))
    boundaries = np.unique(np.concatenate([[0], boundaries, [len(frame_labels)]]))
    return boundaries
