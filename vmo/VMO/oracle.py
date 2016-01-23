"""
oracle.py
Variable Markov Oracle in python

@copyright: 
Copyright (C) 9.2014 Cheng-i Wang

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
import scipy.spatial.distance as dist
import vmo.VMO.utility as utl


class data(object):
    """A helper class to encapsulate objects for symbolic comparison

    By default, the first entry of the list or tuple is used as the feature to
    test for equality between different data object.

    Attributes:
        content: a list or tuple
        idx: the index of the list or tuple to be tested for equality
    """

    def __init__(self, data_item, index=0):
        self.content = data_item
        self.idx = index

    def __repr__(self):
        return str(self.content)

    def __eq__(self, other):
        if type(other) == data:
            if self.content[self.idx] == other.content[self.idx]:
                return True
            else:
                return False
        else:
            return False

    def __ne__(self, other):
        return not (self == other)


class FactorOracle(object):
    """ The base class for the FO(factor oracle) and MO(variable markov oracle)
    
    Attributes:
        sfx: a list containing the suffix link of each state.
        trn: a list containing the forward links of each state as a list.
        rsfx: a list containing the reverse suffix links of each state 
            as a list.
        lrs: the value of longest repeated suffix of each state.
        data: the symobols associated with the direct link 
            connected to each state.
        compror: a list of tuples (i, i-j), i is the current coded position,
            i-j is the length of the corresponding coded words.
        code: a list of tuples (len, pos), len is the length of the 
            corresponding coded words, pos is the position where the coded
            words starts.
        seg: same as code but non-overlapping.
        f_array: (For kind 'a' and 'v'): a list containing the feature array
        latent: (For kind 'a' and 'v'): a list of lists with each sub-list
            containing the indexes for each symbol.
        kind: 
            'a': Variable Markov oracle
            'f': repeat oracle
            'v': Centroid-based oracle (under test)
        n_states: number of total states, also is length of the input 
            sequence plus 1.
        max_lrs: the longest lrs so far.
        avg_lrs: the average lrs so far.
        name: the name of the oracle.
        params: a python dictionary for different feature and distance settings.
            keys:
                'thresholds': the minimum value for separating two values as 
                    different symbols.
                'weights': a dictionary containing different weights for features
                    used.
                'dfunc': the distance function.
    """

    def __init__(self, **kwargs):
        # Basic attributes
        self.sfx = []
        self.trn = []
        self.rsfx = []
        self.lrs = []
        self.data = []

        # Compression attributes
        self.compror = []
        self.code = []
        self.seg = []

        # Object attributes
        self.kind = 'f'
        self.name = ''

        # Oracle statistics
        self.n_states = 1
        self.max_lrs = []
        self.max_lrs.append(0)
        self.avg_lrs = []
        self.avg_lrs.append(0.0)

        # Oracle parameters
        self.params = {
            'threshold': 0,
            'dfunc': 'euclidean',
            'dfunc_handle': None,
            'dim': 1
        }
        self.update_params(**kwargs)

        # Adding zero state
        self.sfx.append(None)
        self.rsfx.append([])
        self.trn.append([])
        self.lrs.append(0)
        self.data.append(0)

    def reset(self, **kwargs):
        self.update_params(**kwargs)
        # Basic attributes
        self.sfx = []
        self.trn = []
        self.rsfx = []
        self.lrs = []
        self.data = []

        # Compression attributes
        self.compror = []
        self.code = []
        self.seg = []

        # Object attributes
        self.kind = 'f'
        self.name = ''

        # Oracle statistics
        self.n_states = 1
        self.max_lrs = []
        self.max_lrs.append(0)
        self.avg_lrs = []
        self.avg_lrs.append(0.0)

        # Adding zero state
        self.sfx.append(None)
        self.rsfx.append([])
        self.trn.append([])
        self.lrs.append(0)
        self.data.append(0)

    def update_params(self, **kwargs):
        """Subclass this"""
        self.params.update(kwargs)

    def add_state(self, new_data):
        """Subclass this"""
        pass

    def _encode(self):
        _code = []
        _compror = []
        if not self.compror:
            j = 0
        else:
            j = self.compror[-1][0]

        i = j
        while j < self.n_states - 1:
            while i < self.n_states - 1 and self.lrs[i + 1] >= i - j + 1:
                i += 1
            if i == j:
                i += 1
                _code.append([0, i])
                _compror.append([i, 0])
            else:
                _code.append([i - j, self.sfx[i] - i + j + 1])
                _compror.append([i, i - j])
            j = i
        return _code, _compror

    def encode(self):
        _c, _cmpr = self._encode()
        self.code.extend(_c)
        self.compror.extend(_cmpr)

        return self.code, self.compror

    @property
    def segment(self):
        """An non-overlap version Compror"""

        if not self.seg:
            j = 0
        else:
            j = self.seg[-1][1]
            last_len = self.seg[-1][0]
            if last_len + j > self.n_states:
                return

        i = j
        while j < self.n_states - 1:
            while not (not (i < self.n_states - 1) or not (self.lrs[i + 1] >= i - j + 1)):
                i += 1
            if i == j:
                i += 1
                self.seg.append((0, i))
            else:
                if (self.sfx[i] + self.lrs[i]) <= i:
                    self.seg.append((i - j, self.sfx[i] - i + j + 1))

                else:
                    _i = j + i - self.sfx[i]
                    self.seg.append((_i - j, self.sfx[i] - i + j + 1))
                    _j = _i
                    while not (not (_i < i) or not (self.lrs[_i + 1] - self.lrs[_j] >= _i - _j + 1)):
                        _i += 1
                    if _i == _j:
                        _i += 1
                        self.seg.append((0, _i))
                    else:
                        self.seg.append((_i - _j, self.sfx[_i] - _i + _j + 1))
            j = i
        return self.seg

    def _ir(self, alpha=1.0):
        code, _ = self.encode()
        cw = np.zeros(len(code))  # Number of code words
        for i, c in enumerate(code):
            cw[i] = c[0] + 1

        c0 = [1 if x[0] == 0 else 0 for x in self.code]
        h0 = np.log2(np.cumsum(c0))

        h1 = np.zeros(len(cw))

        for i in range(1, len(cw)):
            h1[i] = utl.entropy(cw[0:i + 1])

        ir = alpha * h0 - h1

        return ir, h0, h1

    def _ir_fixed(self, alpha=1.0):
        code, _ = self.encode()

        h0 = np.log2(self.num_clusters())

        if self.max_lrs[-1] == 0:
            h1 = np.log2(self.n_states - 1)
        else:
            h1 = np.log2(self.n_states - 1) + np.log2(self.max_lrs[-1])

        BL = np.zeros(self.n_states - 1)
        j = 0
        for i in range(len(code)):
            if self.code[i][0] == 0:
                BL[j] = 1
                j += 1
            else:
                L = code[i][0]
                BL[j:j + L] = L  # range(1,L+1)
                j = j + L
        ir = alpha * h0 - h1 / BL
        ir[ir < 0] = 0
        return ir, h0, h1

    def _ir_cum(self, alpha=1.0):
        code, _ = self.encode()

        N = self.n_states

        cw0 = np.zeros(N - 1)  # cw0 counts the appearance of new states only
        cw1 = np.zeros(N - 1)  # cw1 counts the appearance of all compror states
        BL = np.zeros(N - 1)  # BL is the block length of compror codewords

        j = 0
        for i in range(len(code)):
            if self.code[i][0] == 0:
                cw0[j] = 1
                cw1[j] = 1
                BL[j] = 1
                j += 1
            else:
                L = code[i][0]
                cw1[j] = 1
                BL[j:j + L] = L  # range(1,L+1)
                j = j + L

        h0 = np.log2(np.cumsum(cw0))
        h1 = np.log2(np.cumsum(cw1))
        h1 = h1 / BL
        ir = alpha * h0 - h1
        ir[ir < 0] = 0

        return ir, h0, h1

    def _ir_cum2(self, alpha=1.0):
        code, _ = self.encode()

        N = self.n_states
        BL = np.zeros(N - 1)  # BL is the block length of compror codewords

        h0 = np.log2(np.cumsum(
            [1.0 if sfx == 0 else 0.0 for sfx in self.sfx[1:]])
        )
        """
        h1 = np.array([h if m == 0 else h+np.log2(m) 
                       for h,m in zip(h0,self.lrs[1:])])
        h1 = np.array([h if m == 0 else h+np.log2(m) 
                       for h,m in zip(h0,self.max_lrs[1:])])
        h1 = np.array([h if m == 0 else h+np.log2(m) 
                       for h,m in zip(h0,self.avg_lrs[1:])])
        """
        h1 = np.array([np.log2(i + 1) if m == 0 else np.log2(i + 1) + np.log2(m)
                       for i, m in enumerate(self.max_lrs[1:])])

        j = 0
        for i in range(len(code)):
            if self.code[i][0] == 0:
                BL[j] = 1
                j += 1
            else:
                L = code[i][0]
                BL[j:j + L] = L  # range(1,L+1)
                j = j + L

        h1 = h1 / BL
        ir = alpha * h0 - h1
        ir[ir < 0] = 0  # Really a HACK here!!!!!
        return ir, h0, h1

    def _ir_cum3(self, alpha=1.0):

        h0 = np.log2(np.cumsum(
            [1.0 if sfx == 0 else 0.0 for sfx in self.sfx[1:]])
        )
        h1 = np.array([h if m == 0 else (h + np.log2(m)) / m
                       for h, m in zip(h0, self.lrs[1:])])

        ir = alpha * h0 - h1
        ir[ir < 0] = 0  # Really a HACK here!!!!!
        return ir, h0, h1

    def IR(self, alpha=1.0, ir_type='cum'):
        if ir_type == 'cum':
            return self._ir_cum(alpha)
        elif ir_type == 'all':
            return self._ir(alpha)
        elif ir_type == 'fixed':
            return self._ir_fixed(alpha)
        elif ir_type == 'cum2':
            return self._ir_cum2(alpha)
        elif ir_type == 'cum3':
            return self._ir_cum3(alpha)

    def num_clusters(self):
        return len(self.rsfx[0])

    def threshold(self):
        if self.params.get('threshold'):
            return int(self.params.get('threshold'))
        else:
            raise ValueError("Threshold is not set!")

    def dfunc(self):
        if self.params.get('dfunc'):
            return self.params.get('dfunc')
        else:
            raise ValueError("dfunc is not set!")

    def dfunc_handle(self, a, b_vec):
        fun = self.params['dfunc_handle']
        return fun(a, b_vec)

    def _len_common_suffix(self, p1, p2):
        if p2 == self.sfx[p1]:
            return self.lrs[p1]
        else:
            while self.sfx[p2] != self.sfx[p1] and p2 != 0:
                p2 = self.sfx[p2]
        return min(self.lrs[p1], self.lrs[p2])

    def _find_better(self, i, symbol):
        self.rsfx[i].sort()
        for j in self.rsfx[i]:
            if (self.lrs[j] == self.lrs[i] and
                        self.data[j - self.lrs[i]] == symbol):
                return j
        return None


class FO(FactorOracle):
    """ An implementation of the factor oracle
    """

    def __init__(self, **kwargs):
        super(FO, self).__init__(**kwargs)
        self.kind = 'r'

    def add_state(self, new_symbol):
        """

        :type self: oracle
        """
        self.sfx.append(0)
        self.rsfx.append([])
        self.trn.append([])
        self.lrs.append(0)
        self.data.append(new_symbol)

        self.n_states += 1

        i = self.n_states - 1

        self.trn[i - 1].append(i)
        k = self.sfx[i - 1]
        pi_1 = i - 1

        # Adding forward links
        while k is not None:
            _symbols = [self.data[state] for state in self.trn[k]]
            if self.data[i] not in _symbols:
                self.trn[k].append(i)
                pi_1 = k
                k = self.sfx[k]
            else:
                break

        if k is None:
            self.sfx[i] = 0
            self.lrs[i] = 0
        else:
            _query = [[self.data[state], state] for state in
                      self.trn[k] if self.data[state] == self.data[i]]
            _query = sorted(_query, key=lambda _query: _query[1])
            _state = _query[0][1]
            self.sfx[i] = _state
            self.lrs[i] = self._len_common_suffix(pi_1, self.sfx[i] - 1) + 1

        k = self._find_better(i, self.data[i - self.lrs[i]])
        if k is not None:
            self.lrs[i] += 1
            self.sfx[i] = k
        self.rsfx[self.sfx[i]].append(i)

        if self.lrs[i] > self.max_lrs[i - 1]:
            self.max_lrs.append(self.lrs[i])
        else:
            self.max_lrs.append(self.max_lrs[i - 1])

        self.avg_lrs.append(self.avg_lrs[i - 1] * ((i - 1.0) / (self.n_states - 1.0)) +
                            self.lrs[i] * (1.0 / (self.n_states - 1.0)))

    def accept(self, context):
        """ Check if the context could be accepted by the oracle
        
        Args:
            context: s sequence same type as the oracle data
        
        Returns:
            bAccepted: whether the sequence is accepted or not
            _next: the state where the sequence is accepted
        """
        _next = 0
        for _s in context:
            _data = [self.data[j] for j in self.trn[_next]]
            if _s in _data:
                _next = self.trn[_next][_data.index(_s)]
            else:
                return 0, _next
        return 1, _next

    def get_alphabet(self):
        alphabet = [self.data[i] for i in self.trn[0]]
        dictionary = dict(zip(alphabet, range(len(alphabet))))
        return dictionary


class MO(FactorOracle):
    def __init__(self, **kwargs):
        super(MO, self).__init__(**kwargs)
        self.kind = 'a'
        self.f_array = feature_array(self.params['dim'])
        self.f_array.add(np.zeros(self.params['dim'], ))
        self.data[0] = None
        self.latent = []

    def reset(self, **kwargs):
        super(MO, self).reset(**kwargs)

        self.kind = 'a'
        # self.f_array = [0]
        self.f_array = feature_array(self.params['dim'])
        self.f_array.add(np.zeros(self.params['dim'], ))
        self.data[0] = None
        self.latent = []

    def add_state(self, new_data, method='inc'):
        """Create new state and update related links and compressed state"""
        self.sfx.append(0)
        self.rsfx.append([])
        self.trn.append([])
        self.lrs.append(0)

        # Experiment with pointer-based
        self.f_array.add(new_data)

        self.n_states += 1
        i = self.n_states - 1

        # assign new transition from state i-1 to i
        self.trn[i - 1].append(i)
        k = self.sfx[i - 1]
        pi_1 = i - 1

        # iteratively backtrack suffixes from state i-1
        if method == 'inc':
            suffix_candidate = 0
        elif method == 'complete':
            suffix_candidate = []
        else:
            suffix_candidate = 0

        while k is not None:

            if self.params['dfunc'] == 'other':
                dvec = self.dfunc_handle(new_data,
                                         self.f_array[self.trn[k]])
            else:
                dvec = dist.cdist([new_data],
                                  self.f_array[self.trn[k]],
                                  metric=self.params['dfunc'])[0]

            I = np.where(dvec < self.params['threshold'])[0]
            if len(I) == 0:  # if no transition from suffix
                self.trn[k].append(i)  # Add new forward link to unvisited state
                pi_1 = k
                if method != 'complete':
                    k = self.sfx[k]
            else:
                if method == 'inc':
                    if I.shape[0] == 1:
                        suffix_candidate = self.trn[k][I[0]]
                    else:
                        suffix_candidate = self.trn[k][I[np.argmin(dvec[I])]]
                    break
                elif method == 'complete':
                    suffix_candidate.append((self.trn[k][I[np.argmin(dvec[I])]],
                                             np.min(dvec)))
                else:
                    suffix_candidate = self.trn[k][I[np.argmin(dvec[I])]]
                    break

            if method == 'complete':
                k = self.sfx[k]

        if method == 'complete':
            if not suffix_candidate:
                self.sfx[i] = 0
                self.lrs[i] = 0
                self.latent.append([i])
                self.data.append(len(self.latent) - 1)
            else:
                sorted_suffix_candidates = sorted(suffix_candidate,
                                                  key=lambda suffix: suffix[1])
                self.sfx[i] = sorted_suffix_candidates[0][0]
                self.lrs[i] = self._len_common_suffix(pi_1, self.sfx[i] - 1) + 1
                self.latent[self.data[self.sfx[i]]].append(i)
                self.data.append(self.data[self.sfx[i]])
        else:
            if k is None:
                self.sfx[i] = 0
                self.lrs[i] = 0
                self.latent.append([i])
                self.data.append(len(self.latent) - 1)
            else:
                self.sfx[i] = suffix_candidate
                self.lrs[i] = self._len_common_suffix(pi_1, self.sfx[i] - 1) + 1
                self.latent[self.data[self.sfx[i]]].append(i)
                self.data.append(self.data[self.sfx[i]])

        self.rsfx[self.sfx[i]].append(i)

        if self.lrs[i] > self.max_lrs[i - 1]:
            self.max_lrs.append(self.lrs[i])
        else:
            self.max_lrs.append(self.max_lrs[i - 1])

        self.avg_lrs.append(self.avg_lrs[i - 1] * ((i - 1.0) / (self.n_states - 1.0)) +
                            self.lrs[i] * (1.0 / (self.n_states - 1.0)))


class VMO(FactorOracle):
    def __init__(self, **kwargs):
        super(VMO, self).__init__(**kwargs)
        self.kind = 'v'
        self.f_array = [0]
        self.data[0] = None
        self.latent = []
        self.centroid = []
        self.hist = []

        # including connectivity
        self.con = []
        self.transition = []

    def reset(self, **kwargs):
        super(VMO, self).reset(**kwargs)

        self.kind = 'v'
        self.f_array = [0]
        self.data[0] = None
        self.latent = []
        self.centroid = []
        self.hist = []

        # including connectivity
        self.con = []
        self.transition = []

    def add_state(self, new_data):
        self.sfx.append(0)
        self.rsfx.append([])
        self.trn.append([])
        self.lrs.append(0)

        # Experiment with pointer-based  
        self.f_array.append(new_data)

        self.n_states += 1
        i = self.n_states - 1

        # assign new transition from state i-1 to i
        self.trn[i - 1].append(i)
        k = self.sfx[i - 1]
        pi_1 = i - 1

        dvec = []
        trn_list = []
        suffix_candidate = 0

        while k is not None:
            # NEW Implementation

            if self.params['dfunc'] == 'others':
                dvec = self.dfunc_handle(new_data,
                                         [self.centroid[self.data[t]] for
                                          t in self.trn[k]])
            else:
                dvec = dist.cdist([new_data], np.array([self.centroid[self.data[t]] for
                                                        t in self.trn[k]]),
                                  metric=self.params['dfunc'])[0]

            # if no transition from suffix
            I = np.where(dvec < self.params['threshold'])[0]
            if len(I) == 0:
                self.trn[k].append(i)  # Create a new forward link to unvisited state
                trn_list.append(k)
                pi_1 = k
                k = self.sfx[k]
            else:
                suffix_candidate = self.trn[k][I[np.argmin(dvec[I])]]
                trn_list.append(i - 1)
                break

        if k is None:
            self.sfx[i] = 0
            self.lrs[i] = 0
            self.latent.append([i])
            self.hist.append(1)
            self.data.append(len(self.latent) - 1)
            self.centroid.append(new_data)
            if i > 1:
                self.con[self.data[i - 1]].add(self.data[i])
                self.con.append({self.data[i]})
            else:
                self.con.append(set([]))
        else:
            self.sfx[i] = suffix_candidate
            self.lrs[i] = self._len_common_suffix(pi_1, self.sfx[i] - 1) + 1
            _i = self.data[self.sfx[i]]
            self.latent[_i].append(i)
            self.hist[_i] += 1
            self.data.append(_i)
            self.centroid[_i] = ((self.centroid[_i] * (self.hist[_i] - 1)
                                  + new_data) /
                                 self.hist[_i])
            self.con[self.data[i - 1]].add(self.data[i])
            map(set.add, [self.con[self.data[c]] for c in trn_list],
                [self.data[i]] * len(trn_list))
        self.rsfx[self.sfx[i]].append(i)

        if self.lrs[i] > self.max_lrs[i - 1]:
            self.max_lrs.append(self.lrs[i])
        else:
            self.max_lrs.append(self.max_lrs[i - 1])

        self.avg_lrs.append(self.avg_lrs[i - 1] * ((i - 1.0) / (self.n_states - 1.0)) +
                            self.lrs[i] * (1.0 / (self.n_states - 1.0)))


class feature_array:
    def __init__(self, dim):
        self.data = np.zeros((100, dim))
        self.dim = dim
        self.capacity = 100
        self.size = 0

    def __getitem__(self, item):
        return self.data[item, :]

    def add(self, x):
        if self.size == self.capacity:
            self.capacity *= 4
            newdata = np.zeros((self.capacity, self.dim))
            newdata[:self.size, :] = self.data
            self.data = newdata

        self.data[self.size, :] = x
        self.size += 1

    def finalize(self):
        self.data = self.data[:self.size, :]


def _create_oracle(oracle_type, **kwargs):
    """A routine for creating a factor oracle."""
    if oracle_type == 'f':
        return FO(**kwargs)
    elif oracle_type == 'a':
        return MO(**kwargs)
    elif oracle_type == 'v':
        return VMO(**kwargs)
    else:
        return MO(**kwargs)


def create_oracle(flag, threshold=0, dfunc='euclidean',
                  dfunc_handle=None, dim=1):
    return _create_oracle(flag, threshold=threshold, dfunc=dfunc,
                          dfunc_handle=dfunc_handle, dim=dim)


def _build_oracle(flag, oracle, input_data, suffix_method='inc'):
    if type(input_data) != np.ndarray or type(input_data[0]) != np.ndarray:
        input_data = np.array(input_data)

    if input_data.ndim != 2:
        input_data = np.expand_dims(input_data, axis=1)

    if flag == 'a':
        [oracle.add_state(obs, suffix_method) for obs in input_data]
        # for obs in input_data:
        #     oracle.add_state(obs, suffix_method)
        oracle.f_array.finalize()
    else:
        [oracle.add_state(obs) for obs in input_data]
        # for obs in input_data:
        #     oracle.add_state(obs)
    return oracle


def build_oracle(input_data, flag,
                 threshold=0, suffix_method='inc',
                 feature=None, weights=None, dfunc='cosine',
                 dfunc_handle=None, dim=1):
    # initialize weights if needed1
    if weights is None:
        weights = {}
        weights.setdefault(feature, 1.0)

    if flag == 'a':
        oracle = _create_oracle(flag, threshold=threshold, dfunc=dfunc,
                                dfunc_handle=dfunc_handle, dim=dim)
        oracle = _build_oracle(flag, oracle, input_data, suffix_method)
    elif flag == 'f' or flag == 'v':
        oracle = _create_oracle(flag, threshold=threshold, dfunc=dfunc,
                                dfunc_handle=dfunc_handle, dim=dim)
        oracle = _build_oracle(flag, oracle, input_data)
    else:
        oracle = _create_oracle('a', threshold=threshold, dfunc=dfunc,
                                dfunc_handle=dfunc_handle, dim=dim)
        oracle = _build_oracle(flag, oracle, input_data, suffix_method)

    return oracle


def find_threshold_sgd(input_data, r=(0, 1, 0.1), method='ir', flag='a',
                       suffix_method='inc', alpha=1.0, feature=None, ir_type='cum',
                       dfunc='cosine', dfunc_handle=None, dim=1):
    thresholds = np.arange(r[0], r[1], r[2])
    prev_ir = 0
    for t in thresholds:
        tmp_oracle = build_oracle(input_data, flag=flag, threshold=t,
                                  suffix_method=suffix_method, feature=feature,
                                  dfunc=dfunc, dfunc_handle=dfunc_handle, dim=dim)
        tmp_ir, h0, h1 = tmp_oracle.IR(ir_type=ir_type, alpha=alpha)
        ir = tmp_ir.sum()
        if ir < prev_ir:
            return t - r[2], prev_ir
        else:
            prev_ir = ir


def find_threshold_nt(input_data, r=(0, 1, 0.1), method='ir', flag='a',
                      suffix_method='inc', alpha=1.0, feature=None, ir_type='cum',
                      dfunc='cosine', dfunc_handle=None, dim=1):
    t = r[1] / 2.0
    tmp_oracle = build_oracle(input_data, flag=flag, threshold=t,
                              suffix_method=suffix_method, feature=feature,
                              dfunc=dfunc, dfunc_handle=dfunc_handle, dim=dim)
    tmp_ir, h0, h1 = tmp_oracle.IR(ir_type=ir_type, alpha=alpha)
    ir = tmp_ir.sum()

    t_tmp = t + r[2]
    tmp_oracle = build_oracle(input_data, flag=flag, threshold=t_tmp,
                              suffix_method=suffix_method, feature=feature,
                              dfunc=dfunc, dfunc_handle=dfunc_handle, dim=dim)
    tmp_ir, h0, h1 = tmp_oracle.IR(ir_type=ir_type, alpha=alpha)
    ir_tmp = tmp_ir.sum()

    if ir_tmp > ir:
        while ir_tmp > ir:
            t = t_tmp
            ir = ir_tmp

            t_tmp = t + r[2]
            tmp_oracle = build_oracle(input_data, flag=flag, threshold=t_tmp,
                                      suffix_method=suffix_method, feature=feature,
                                      dfunc=dfunc, dfunc_handle=dfunc_handle, dim=dim)
            tmp_ir, h0, h1 = tmp_oracle.IR(ir_type=ir_type, alpha=alpha)
            ir_tmp = tmp_ir.sum()
    else:
        t_tmp = t - r[2]
        tmp_oracle = build_oracle(input_data, flag=flag, threshold=t_tmp,
                                  suffix_method=suffix_method, feature=feature,
                                  dfunc=dfunc, dfunc_handle=dfunc_handle, dim=dim)
        tmp_ir, h0, h1 = tmp_oracle.IR(ir_type=ir_type, alpha=alpha)
        ir_tmp = tmp_ir.sum()

        while ir_tmp > ir:
            t = t_tmp
            ir = ir_tmp

            t_tmp = t - r[2]
            tmp_oracle = build_oracle(input_data, flag=flag, threshold=t_tmp,
                                      suffix_method=suffix_method, feature=feature,
                                      dfunc=dfunc, dfunc_handle=dfunc_handle, dim=dim)
            tmp_ir, h0, h1 = tmp_oracle.IR(ir_type=ir_type, alpha=alpha)
            ir_tmp = tmp_ir.sum()
    return t, ir


def find_threshold(input_data, r=(0, 1, 0.1), method='ir', flag='a',
                   suffix_method='inc', alpha=1.0, feature=None, ir_type='cum',
                   dfunc='cosine', dfunc_handle=None, dim=1,
                   verbose=False, entropy=False):
    if method == 'ir':
        return find_threshold_ir(input_data, r, flag, suffix_method, alpha,
                                 feature, ir_type, dfunc, dfunc_handle, dim,
                                 verbose, entropy)
    # elif method == 'motif':
    #     return find_threshold_motif(input_data, r, flag, suffix_method, alpha,
    #                                 feature, dfunc, dfunc_handle, dim, verbose)


def find_threshold_ir(input_data, r=(0, 1, 0.1), flag='a', suffix_method='inc',
                      alpha=1.0, feature=None, ir_type='cum',
                      dfunc='cosine', dfunc_handle=None, dim=1,
                      verbose=False, entropy=False):
    thresholds = np.arange(r[0], r[1], r[2])
    irs = []
    if entropy:
        h0_vec = []
        h1_vec = []
    for t in thresholds:
        if verbose:
            print 'Testing threshold:', t
        tmp_oracle = build_oracle(input_data, flag=flag, threshold=t,
                                  suffix_method=suffix_method, feature=feature,
                                  dfunc=dfunc, dfunc_handle=dfunc_handle, dim=dim)
        tmp_ir, h0, h1 = tmp_oracle.IR(ir_type=ir_type, alpha=alpha)
        irs.append(tmp_ir.sum())
        if entropy:
            h0_vec.append(h0.sum())
            h1_vec.append(h1.sum())
    # now pair irs and thresholds in a vector, and sort by ir
    ir_thresh_pairs = [(a, b) for a, b in zip(irs, thresholds)]
    pairs_return = ir_thresh_pairs
    ir_thresh_pairs = sorted(ir_thresh_pairs, key=lambda x: x[0],
                             reverse=True)
    if entropy:
        return ir_thresh_pairs[0], pairs_return, h0_vec, h1_vec
    else:
        return ir_thresh_pairs[0], pairs_return

# def find_threshold_motif(input_data, r=(0, 1, 0.1), flag='a',
#                          suffix_method='inc', alpha=1.0, feature=None,
#                          dfunc='euclidean', dfunc_handle=None, dim=1,
#                          verbose=False):
#     thresholds = np.arange(r[0], r[1], r[2])
#     avg_len = []
#     avg_occ = []
#     avg_num = []
#
#     for t in thresholds:
#         tmp_oracle = build_oracle(input_data, flag=flag, threshold=t,
#                                   suffix_method=suffix_method, feature=feature,
#                                   dfunc=dfunc, dfunc_handle=dfunc_handle, dim=dim)
#         pttr = van.find_repeated_patterns(tmp_oracle, alpha)
#         if not pttr:
#             avg_len.append(np.mean([float(p[1]) for p in pttr]))
#             avg_occ.append(np.mean([float(len(p[0])) for p in pttr]))
#             avg_num.append(len(pttr))
#         else:
#             avg_len.append(0.0)
#             avg_occ.append(0.0)
#             avg_num.append(0.0)
#         if verbose:
#             print 'Testing threshold:', t
#             print '          avg_len:', avg_len[-1]
#             print '          avg_occ:', avg_occ[-1]
#             print '          avg_num:', avg_num[-1]
#
#     return avg_len, avg_occ, avg_num, thresholds
