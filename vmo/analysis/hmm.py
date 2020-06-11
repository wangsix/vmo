"""hmm.py
offline factor/variable markov oracle generation routines for vmo

Copyright (C) 2.16.2017 Cheng-i Wang

This file is part of vmo.

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
"""

import numpy as np


def extract_hmm_tensor(oracle, max_lrs=None, normalize=True, smooth=False):
    n = oracle.num_clusters()

    if max_lrs is None:
        max_lrs = np.max(oracle.max_lrs)-1
    if smooth:
        hmm_tensor = np.ones((max_lrs, n, n))
    else:
        hmm_tensor = np.zeros((max_lrs, n, n))

    for r, d, e in zip(oracle.lrs[:1:-1], oracle.data[:1:-1], oracle.data[-2:0:-1]):
        if r <= 2:
            hmm_tensor[:1, e, d] += 1.0
        else:
            hmm_tensor[:r - 1, e, d] += 1.0

    if normalize:
        for i, m in enumerate(hmm_tensor):
            # m += np.finfo('float').eps
            divider = np.sum(m, axis=1)
            divider[divider == 0.0] = 1.0
            hmm_tensor[i] = np.divide(m.T, divider).T

    return hmm_tensor


def recognition(obs, oracle, order=1, smooth=False):
    hmm_tensor = extract_hmm_tensor(oracle, max_lrs=order, smooth=smooth)

    cluster_means = np.array([np.median(oracle.f_array.data[np.array(c), :].T, axis=1)
                              for c in oracle.latent])

    cluster_means += np.finfo('float').eps
    cluster_means = (cluster_means.T / np.sum(cluster_means, axis=1)).T

    a = hmm_tensor[-1]
    a += np.finfo('float').eps
    a += 1.0
    divider = np.sum(a, axis=1)
    a = np.divide(a.T, divider).T
    log_a = np.log(a)
    # hist = np.array([len(c) for c in oracle.latent])/float(oracle.n_states-1)

    v = np.zeros((len(obs), len(oracle.latent)))
    p = np.zeros(v.shape)
    v[0] = np.log(np.dot(cluster_means, obs[0])) + np.log(1.0/len(oracle.latent))
    # v[0] = np.log(np.dot(cluster_means, obs[0])) + np.log(hist)
    # p[0] = np.arange(len(oracle.latent))
    for t in range(1, len(obs)):
        s = v[t-1]+log_a.T
        v[t] = np.max(s, axis=1)+np.log(np.dot(cluster_means, obs[t]))
        p[t-1] = np.argmax(s, axis=1)

    return v, p
