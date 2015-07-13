"""analysis.py
offline factor/variable markov oracle generation routines for vmo

Copyright (C) 7.28.2014 Cheng-i Wang

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

import vmo, sys, itertools, librosa
import numpy as np
import scipy.spatial.distance as dist
import scipy.cluster.hierarchy as scihc
import scipy.signal as sig
import sklearn.cluster as sklhc
from functools import partial
from scipy.stats import multivariate_normal
import fuzzywuzzy.fuzz as fuzz
import vmo.VMO
import vmo.VMO.utility as utils

'''Self-similarity matrix and transition matrix from an oracle
'''


def create_selfsim(oracle, method='rsfx'):
    """ Create self similarity matrix from compror codes or suffix links
    
    :type oracle: a vmo object
    Args:
        oracle: a encoded vmo object
        method: 
            "comp" - use the compression codes
            "sfx" - use suffix links
            "rsfx" - use reverse suffix links
            "lrs" - use LRS values
            "pttr" - use patterns found
        
    """
    len_oracle = oracle.n_states - 1
    mat = np.zeros((len_oracle, len_oracle))
    if method == 'com':
        if not oracle.code:
            print "Codes not generated. Generating codes with encode()."
            oracle.encode()
        ind = 0  # index
        for l, p in oracle.code:  # l for length, p for position

            if l == 0:
                inc = 1
            else:
                inc = l
            mat[range(ind, ind + inc), range(p - 1, p - 1 + inc)] = 1
            mat[range(p - 1, p - 1 + inc), range(ind, ind + inc)] = 1
            ind = ind + l
    elif method == 'sfx':
        for i, s in enumerate(oracle.sfx[1:]):
            if s != 0:
                mat[i][s - 1] = 1
                mat[s - 1][i] = 1
    elif method == 'rsfx':
        for cluster in oracle.latent:
            p = itertools.product(cluster, repeat=2)
            for _p in p:
                mat[_p[0] - 1][_p[1] - 1] = 1
    elif method == 'lrs':
        for i, l in enumerate(oracle.lrs[1:]):
            if l != 0:
                s = oracle.sfx[i + 1]
                mat[range((s - l) + 1, s + 1), range(i - l + 1, i + 1)] = 1
                mat[range(i - l + 1, i + 1), range((s - l) + 1, s + 1)] = 1
    elif method == 'seg':
        seg = oracle.segment
        ind = 0
        for l, p in seg:  # l for length, p for position

            if l == 0:
                inc = 1
            else:
                inc = l
            mat[range(ind, ind + inc), range(p - 1, p - 1 + inc)] = 1
            mat[range(p - 1, p - 1 + inc), range(ind, ind + inc)] = 1
            ind = ind + l

    return mat


def create_transition(oracle, method='trn'):
    """Create a transition matrix based on oracle links"""
    mat, hist, n = _create_trn_mat_symbolic(oracle, method)
    return mat, hist, n


def _create_trn_mat_symbolic(oracle, method):
    trn_list = None
    n = oracle.num_clusters()
    sym_list = [oracle.data[_s] for _s in oracle.rsfx[0]]
    hist = np.zeros(n, )
    mat = np.zeros((n, n))
    for i in range(1, oracle.n_states - 1):
        _i = sym_list.index(oracle.data[i])
        if method == 'trn':
            trn_list = oracle.trn[i]
        elif method == 'seq':
            trn_list = [i + 1]

        for j in trn_list:
            if j < oracle.n_states:
                _j = sym_list.index(oracle.data[j])
                mat[_i][_j] += 1
            else:
                print "index " + str(j) + " is out of bounds."
            hist[_i] += 1
    mat = mat.transpose() / hist
    mat = mat.transpose()
    return mat, hist, n


'''
Symbolic sequence prediction by an oracle
'''


def predict(oracle, context, ab=None, verbose=False):
    if verbose:
        print "original context: ", context
    if ab is None:
        ab = oracle.get_alphabet()

    _b, _s, context = _test_context(oracle, context)
    _lrs = [oracle.lrs[k] for k in oracle.rsfx[_s]]
    context_state = []
    while not context_state:
        for _i, _l in enumerate(_lrs):
            if _l >= len(context):
                context_state.append(oracle.rsfx[_s][_i])
        if context_state:
            break
        else:
            context = context[1:]
            _b, _s = oracle.accept(context)
            _lrs = [oracle.lrs[k] for k in oracle.rsfx[_s]]
    if verbose:
        print "final context: ", context
        print "context_state: ", context_state
    d_count = len(ab)
    hist = [1.0] * len(ab)  # initialize all histograms with 1s.

    trn_data = [oracle.data[n] for n in oracle.trn[_s]]
    for k in trn_data:
        hist[ab[k]] += 1.0
        d_count += 1.0

    for i in context_state:
        d_count, hist = _rsfx_count(oracle, i, d_count, hist, ab)

    return [hist[idx] / d_count for idx in range(len(hist))], context


def logEval(oracle, testSequence, ab=[], m_order=None, VERBOSE=False):
    """ Evaluate the average log-loss of a sequence given an oracle """

    if not ab:
        ab = oracle.get_alphabet()
    if VERBOSE:
        print ' '

    logP = 0.0
    context = []
    increment = np.floor((len(testSequence) - 1) / 100)
    bar_count = -1
    maxContextLength = 0
    avgContext = 0
    for i, t in enumerate(testSequence):

        p, c = predict(oracle, context, ab, VERBOSE=False)
        if len(c) < len(context):
            context = context[-len(c):]
        logP -= np.log2(p[ab[t]])
        context.append(t)

        if m_order != None:
            if len(context) > m_order:
                context = context[-m_order:]
        avgContext += float(len(context)) / len(testSequence)

        if VERBOSE:
            percentage = np.mod(i, increment)
            if percentage == 0:
                bar_count += 1
            if len(context) > maxContextLength:
                maxContextLength = len(context)
            sys.stdout.write('\r')
            sys.stdout.write("\r[" + "=" * bar_count +
                             " " * (100 - bar_count) + "] " +
                             str(bar_count) + "% " +
                             str(i) + "/" + str(len(testSequence) - 1) + " Current max length: " + str(
                maxContextLength))
            sys.stdout.flush()
    return logP / len(testSequence), avgContext


def _test_context(oracle, context):
    _b, _s = oracle.accept(context)
    while not _b:
        context = context[1:]
        _b, _s = oracle.accept(context)
    return _b, _s, context


def _rsfx_count(oracle, s, count, hist, ab, VERBOSE=False):
    """ Accumulate counts for context """

    trn_data = [oracle.data[n] for n in oracle.trn[s]]
    for k in trn_data:
        hist[ab[k]] += 1.0
        count += 1.0

    rsfx_candidate = oracle.rsfx[s][:]
    while rsfx_candidate:
        s = rsfx_candidate.pop(0)
        trn_data = [oracle.data[n] for n in oracle.trn[s]]
        for k in trn_data:
            hist[ab[k]] += 1.0
            count += 1.0
        rsfx_candidate.extend(oracle.rsfx[s])

    return count, hist


"""Segmentation algorithms
"""


def _seg_by_single_frame(oracle, cluster_method='agglomerative', connectivity='temporal', data='symbol', width=5, **kwargs):
    obs_len = oracle.n_states - 1
    if connectivity is 'temporal':
        connectivity = np.zeros((obs_len, obs_len))
    else:
        connectivity = create_selfsim(oracle, method=connectivity)
        connectivity = librosa.segment.recurrence_to_lag(connectivity, pad=False)
        connectivity = np.pad(connectivity, [(0, 0), [width, width]], mode='reflect')
        connectivity = sig.medfilt(connectivity, [1, width])
        connectivity = connectivity[:, width:-width]
        connectivity = librosa.segment.lag_to_recurrence(connectivity)

    connectivity[range(1, obs_len), range(obs_len - 1)] = 1.0
    connectivity[range(obs_len - 1), range(1, obs_len)] = 1.0
    connectivity[np.diag_indices(obs_len)] = 0

    if data is 'raw':
        data = np.array(oracle.f_array[1:])
    else:
        data = np.zeros((oracle.n_states - 1, oracle.num_clusters()))
        data[range(oracle.n_states - 1), oracle.data[1:]] = 1

    if cluster_method is 'agglomerative':
        return _seg_by_hc_single_frame(obs_len=obs_len, connectivity=connectivity, data=data, **kwargs)
    elif cluster_method is 'spectral':
        return _seg_by_spectral_single_frame(connectivity)


def _seg_by_hc_single_frame(obs_len, connectivity, data, **kwargs):

    _children, _n_c, _n_leaves, parents, distances = \
        sklhc.ward_tree(data, connectivity=connectivity, return_distance=True)

    reconstructed_z = np.zeros((obs_len - 1, 4))
    reconstructed_z[:, :2] = _children
    reconstructed_z[:, 2] = distances

    if 'threshold' in kwargs.keys():
        t = kwargs['threshold']
    else:
        t = 0.7 * np.max(reconstructed_z[:, 2])

    if 'criterion' in kwargs.keys():
        criterion = kwargs['criterion']
    else:
        criterion = 'distance'

    label = scihc.fcluster(reconstructed_z, t=t, criterion=criterion)

    boundaries = utils.find_boundaries(label)
    labels = utils.segment_labeling(data, boundaries, np.max(label))

    return boundaries, labels


def _seg_by_spectral_single_frame(connectivity):
    graph_lap = utils.normalized_graph_laplacian(connectivity)
    eigen_vecs = utils.eigen_decomposition(graph_lap)
    boundaries, labels = utils.clustering_by_entropy(eigen_vecs, k_min=2)

    return boundaries, labels


def _seg_by_hc_string_matching(oracle):
    pass


def segmentation(oracle, method='symbol_agglomerative', **kwargs):
    if method is 'symbol_agglomerative':
        return _seg_by_single_frame(oracle, cluster_method='agglomerative', **kwargs)
    elif method is 'symbol_spectral':
        return _seg_by_single_frame(oracle, cluster_method='spectral', **kwargs)


"""Query-matching and gesture tracking algorithms"""


def query_complete(oracle, query, trn_type=1, smooth=False, weight=0.5):
    """ Return the closest path in target oracle given a query sequence
    
    Args:
        oracle: an oracle object already learned, the target. 
        query: the query sequence in a matrix form such that 
             the ith row is the feature at the ith time point
        method: 
        trn_type:
        smooth:(off-line only)
        weight:
    
    """
    N = len(query)
    K = oracle.num_clusters()
    P = [[0] * K for _i in range(N)]
    if smooth:
        D = dist.pdist(oracle.f_array[1:], 'sqeuclidean')
        D = dist.squareform(D, checks=False)
        map_k_outer = partial(_query_k, oracle=oracle, query=query, smooth=smooth, D=D, weight=weight)
    else:
        map_k_outer = partial(_query_k, oracle=oracle, query=query)

    map_query = partial(_query_init, oracle=oracle, query=query[0])
    P[0], C = zip(*map(map_query, oracle.rsfx[0][:]))
    P[0] = list(P[0])
    C = np.array(C)

    if trn_type == 1:
        trn = _create_trn_self
    elif trn_type == 2:
        trn = _create_trn_sfx_rsfx
    else:
        trn = _create_trn

    argmin = np.argmin
    distance_cache = np.zeros(oracle.n_states)
    for i in xrange(1, N):  # iterate over the rest of query
        state_cache = []
        dist_cache = distance_cache

        map_k_inner = partial(map_k_outer, i=i, P=P, trn=trn, state_cache=state_cache, dist_cache=dist_cache)
        P[i], _c = zip(*map(map_k_inner, range(K)))
        P[i] = list(P[i])
        C += np.array(_c)

    i_hat = argmin(C)
    P = map(list, zip(*P))
    return P, C, i_hat


def tracking(oracle, obs, trn_type=1, reverse_init=False, method='else', decay=1.0):
    """ Off-line tracking function using sub-optimal query-matching algorithm"""
    N = len(obs)
    if reverse_init:
        r_oracle = create_reverse_oracle(oracle)
        _ind = [r_oracle.n_states - rsfx for rsfx in r_oracle.rsfx[0][:]]
        init_ind = []
        for i in _ind:
            s = i
            while oracle.sfx[s] != 0:
                s = oracle.sfx[s]
            init_ind.append(s)
        K = r_oracle.num_clusters()
    else:
        init_ind = oracle.rsfx[0][:]
        K = oracle.num_clusters()

    P = np.zeros((N, K), dtype='int')
    T = np.zeros((N,), dtype='int')
    map_k_outer = partial(_query_k, oracle=oracle, query=obs)
    map_query = partial(_query_init, oracle=oracle, query=obs[0], method=method)
    #     map_query = partial(_query_init, oracle=oracle, query=obs[0], method)

    argmin = np.argmin

    P[0], C = zip(*map(map_query, init_ind))
    C = np.array(C)
    T[0] = P[0][argmin(C)]

    if trn_type == 1:
        trn = _create_trn_self
    elif trn_type == 2:
        trn = _create_trn_sfx_rsfx
    else:
        trn = _create_trn

    distance_cache = np.zeros(oracle.n_states)

    for i in xrange(1, N):  # iterate over the rest of query
        state_cache = []
        dist_cache = distance_cache

        map_k_inner = partial(map_k_outer, i=i, P=P, trn=trn, state_cache=state_cache, dist_cache=dist_cache)
        P[i], _c = zip(*map(map_k_inner, range(K)))
        C = decay * C + np.array(_c)
        T[i] = P[i][argmin(C)]

    return T


def tracking_multiple_seq(oracle_vec, obs, selftrn=True):
    N = len(obs)  # Length of observation
    K = len(oracle_vec)  # Number of gesture candidates

    P = np.ones((N, K), dtype='int')  # Path matrix
    C = np.zeros((K,))  # Cost vector
    T = np.zeros((N,), dtype='int')  # Tracking index vector
    G = np.zeros((N,), dtype='int')  # Tracking gesture vector

    if selftrn:
        trn = _create_trn_self
    else:
        trn = _create_trn

    for i, _obs in enumerate(obs):
        for k, vo in enumerate(oracle_vec):
            if i == 0:
                a = np.subtract(_obs, vo.f_array[1])
                C[k] += (a * a).sum()
            else:
                s = P[i - 1][k]
                _trn = trn(vo, s)
                dvec = _dist_obs_oracle(vo, _obs, _trn)
                C[k] += np.min(dvec)
                P[i][k] = _trn[np.argmin(dvec)]
        g = np.argmin(C)
        T[i] = P[i][g]
        G[i] = g
    return T, G


def align(oracle, obs, trn_type=1, method='else'):
    N = len(obs)
    init_ind = [1]
    K = 1

    P = np.zeros((N, 1), dtype='int')
    map_k_outer = partial(_query_k, oracle=oracle, query=obs)
    map_query = partial(_query_init, oracle=oracle, query=obs[0], method=method)
    #     map_query = partial(_query_init, oracle=oracle, query=obs[0], method)

    argmin = np.argmin
    P[0], _C = zip(*map(map_query, init_ind))

    if trn_type == 1:
        trn = _create_trn_self
    elif trn_type == 2:
        trn = _create_trn_sfx_rsfx
    else:
        trn = _create_trn

    distance_cache = np.zeros(oracle.n_states)

    for i in xrange(1, N):  # iterate over the rest of query
        state_cache = []
        dist_cache = distance_cache

        map_k_inner = partial(map_k_outer, i=i, P=P, trn=trn,
                              state_cache=state_cache, dist_cache=dist_cache)
        P[i], _c = zip(*map(map_k_inner, range(K)))

    return P


def create_pttr_vmo(oracle, pattern):
    thresh = oracle.params['threshold']

    _vmo_vec = []
    gesture_vmo_vec = []
    for p in pattern:
        _vmo_vec.append([])
        for sfx in p[0]:
            local_obs = oracle.f_array[sfx - p[1] + 1:sfx + 1]
            local_vmo = vmo.build_oracle(local_obs, flag='a', threshold=thresh)
            _vmo_vec[-1].append(local_vmo)

        pttr_vmo = _vmo_vec[-1][0]
        for i in range(pttr_vmo.n_states - 1):
            for mo in _vmo_vec[-1][1:]:
                pttr_vmo.trn[i].extend(set(mo.trn[i]).difference(pttr_vmo.trn[i]))
        gesture_vmo_vec.append(pttr_vmo)

    return gesture_vmo_vec


def query(oracle, query):
    if oracle.kind == 'a':
        mean = [np.mean([oracle.f_array[i] for i in la], axis=0) for la in oracle.latent]
    elif oracle.kind == 'v':
        mean = oracle.centroid[:]

    tran_mat, hist = create_transition(oracle)
    hist = hist / hist.sum()

    N = len(query)
    K = oracle.num_clusters()
    covariance = [np.cov([oracle.f_array[i] for i in la], rowvar=0) for la in oracle.latent]
    rv = [multivariate_normal(mean[i], covariance[i]) for i in range(K)]
    C = np.zeros(K)
    A = np.zeros((N, K))
    L = np.zeros(N)
    # Initialization
    for k in range(K):
        A[0][k] = hist[k] * rv[k].pdf(query[0])
    L[0] = A[0].sum()
    # Induction
    for i in range(1, N):
        for k in range(K):
            A[i][k] = (A[i - 1] * tran_mat[k]).sum() * rv[k].pdf(query[i])
        L[i] = A[i].sum()

    return A, L


def create_reverse_oracle(oracle):
    reverse_data = oracle.f_array[-1:0:-1]
    r_oracle = vmo.build_oracle(reverse_data, 'v', threshold=oracle.params['threshold'])
    return r_oracle


def _query_init(k, oracle, query, method='all'):
    """A helper function for query-matching function initialization."""
    if method == 'all':
        a = np.subtract(query, [oracle.f_array[t] for t in oracle.latent[oracle.data[k]]])
        dvec = (a * a).sum(axis=1)  # Could skip the sqrt
        _d = dvec.argmin()
        return oracle.latent[oracle.data[k]][_d], dvec[_d]

    else:
        a = np.subtract(query, oracle.f_array[k])
        dvec = (a * a).sum()  # Could skip the sqrt
        return k, dvec


def _dist_obs_oracle(oracle, query, trn_list):
    """A helper function calculating distances between a feature and frames in oracle."""
    a = np.subtract(query, [oracle.f_array[t] for t in trn_list])
    return (a * a).sum(axis=1)


def _query_k(k, i, P, oracle, query, trn, state_cache, dist_cache, smooth=False, D=None, weight=0.5):
    """A helper function for query-matching function`s iteration over observations.
    
    Args:
        k - index of the candidate path
        i - index of the frames of the observations
        P - the path matrix of size K x N, K the number for paths initiated, 
            N the frame number of observations
        oracle - an encoded oracle
        query - observations matrix (numpy array) of dimension N x D. 
                D the dimension of the observation.
        trn - function handle of forward links vector gathering
        state_cache - a list storing the states visited during the for loop for k
        dist_cache - a list of the same lenth as oracle storing the 
                    distance calculated between the current observation and states 
                    in the oracle
        smooth - whether to enforce a preference on continuation or not
        D - Self-similarity matrix, required if smooth is set to True
        weight - the weight between continuation or jumps (1.0 for certain continuation)
    
    """

    _trn = trn(oracle, P[i - 1][k])
    t = list(itertools.chain.from_iterable([oracle.latent[oracle.data[j]] for j in _trn]))
    _trn_unseen = [_t for _t in _trn if _t not in state_cache]
    state_cache.extend(_trn_unseen)

    if _trn_unseen:
        t_unseen = list(itertools.chain.from_iterable([oracle.latent[oracle.data[j]] for j in _trn_unseen]))
        dist_cache[t_unseen] = _dist_obs_oracle(oracle, query[i], t_unseen)
    dvec = dist_cache[t]
    if smooth and P[i - 1][k] < oracle.n_states - 1:
        dvec = dvec * (1.0 - weight) + weight * np.array([D[P[i - 1][k]][_t - 1] for _t in t])
    _m = np.argmin(dvec)
    return t[_m], dvec[_m]


def _create_trn_complete(oracle, prev):
    return list(itertools.chain.from_iterable([oracle.latent[_c] for _c in list(oracle.con[oracle.data[prev]])]))


def _create_trn_self(oracle, prev):
    _trn = oracle.trn[prev][:]  # Sub-optimal
    if not _trn:
        _trn = oracle.trn[oracle.sfx[prev]][:]
    _trn.append(prev)
    return _trn


def _create_trn_sfx_rsfx(oracle, prev):
    _trn = oracle.trn[prev][:]
    if not _trn:
        _trn = oracle.trn[oracle.sfx[prev]][:]
        # prev = oracle.sfx[prev]
    else:
        if oracle.rsfx[prev]:
            _trn.extend(oracle.trn[np.min(oracle.rsfx[prev])][:])
        _trn.extend(oracle.trn[oracle.sfx[prev]][:])

    return _trn


def _create_trn(oracle, prev):
    _trn = oracle.trn[prev][:]  # Sub-optimal
    if not _trn:
        _trn = oracle.trn[oracle.sfx[prev]][:]
    return _trn


def _dist2prob(f, a):
    return np.exp(-f / a)


'''Pattern/motif/gesture extraction algorithms
'''


def find_repeated_patterns(oracle, lower=1, inexact=False):
    if lower < 1:
        lower = 1

    pattern_list = []
    prev_sfx = -1
    for i in range(oracle.n_states - 1, lower + 1, -1):
        # Searching back from the end to the last possible position for repeated patterns
        sfx = oracle.sfx[i]
        rsfx = oracle.rsfx[i]
        pattern_found = False
        if (sfx != 0  # not pointing to zeroth state
            and i - oracle.lrs[i] + 1 > sfx and oracle.lrs[i] > lower):  # constraint on length of patterns
            for p in pattern_list:  # for existing pattern
                if not [_p for _p in p[0] if _p - p[1] < i < _p]:
                    if sfx in p[0]:
                        p[0].append(i)
                        lrs_len = np.min([p[1], oracle.lrs[i]])
                        p[1] = lrs_len
                        pattern_found = True
                        break
                    else:
                        pattern_found = False
            if prev_sfx - sfx != 1 and not pattern_found:
                _rsfx = np.array(rsfx).tolist()
                if _rsfx:
                    _rsfx.extend([i, sfx])
                    _len = np.array(oracle.lrs)[_rsfx[:-1]].min()
                    if _len > lower:
                        pattern_list.append([_rsfx, _len])
                else:
                    pattern_list.append([[i, sfx], oracle.lrs[i]])
            prev_sfx = sfx
        else:
            prev_sfx = -1
    return pattern_list


'''
Helper functions
'''


def _get_sfx(oracle, s_set, k):
    while oracle.sfx[k] != 0:
        s_set.add(oracle.sfx[k])
        k = oracle.sfx[k]
    return s_set


def _get_rsfx(oracle, rs_set, k):
    if not oracle.rsfx[k]:
        return rs_set
    else:
        rs_set = rs_set.union(oracle.rsfx[k])
        for _k in oracle.rsfx[k]:
            rs_set = rs_set.union(_get_rsfx(oracle, rs_set, _k))
        return rs_set
