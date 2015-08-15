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

import vmo.VMO.utility as vutl
import vmo.distances as distances

class FactorOracle(object):
    """The base class for the FO(factor oracle) and MO(variable markov oracle)
    
    Attributes:
        sfx: list of ints
            The suffix link of each state.
        trn: list of lists of ints
            The forward links of each state, as a list.
        rsfx: list of lists of ints
            The reverse suffix links of each state, as a list.
        lrs: list of ints
            The length of the longest repeated suffix of each state.
        symbol: list of ints
            The symbol associated with the direct link connected to each state.
        compror: a list of tuples (i, i-j)
            i is the current coded position,
            i-j is the length of the corresponding coded words.
        code: list of tuples (len, pos)
            len is the length of the corresponding coded words,
            pos is the position where the coded words starts.
        seg: list of tuples (len, pos)
            same as code but non-overlapping.
        feature: list of features (for kinds 'a' and 'v')
            Contains the features used to generate the oracle
        latent: a list of lists (for kinds 'a' and 'v')
            latent[i] contains the indexes in the oracle for the i-th symbol
        kind: character
            'a': Variable Markov oracle
            'f': repeat oracle
            'v': Centroid-based oracle (under test)
        n_states: int
            Total number of states. (The length of the input sequence plus 1.)
        max_lrs: int
            The maximum length of a repeated suffix so far.
        avg_lrs: float
            The average length of repeated suffixes so far.
        name: string
            The oracle's name.
        params: dictionary
            Different feature and distance settings.
            keys:
                'threshold': the minimum value for separating two features as 
                    different symbols.
                'weights': a dictionary containing different weights for features
                    used.
                'dfunc': the distance type
                    Supports 'euclidean' as the default and
                             'tonnetz' for a simple Tonnetz-model
                    'other' for user defined
                'dfunc_handle': user-defined distance function
                    (in case of 'other' distance type)
                    Operates on a vector a and a matrix m, returns the vector of
                    distances between a and each column of m.
    """
    def __init__(self, **kwargs):
        # Oracle default parameters
        self.params = {
            'threshold':0,
            'dfunc': 'euclidean',
            'dfunc_handle':None,
            'dim': 1
        }
        self.reset(**kwargs)
        
    def reset(self, **kwargs):
        self.update_params(**kwargs)

        # Basic attributes, initialized with state zero
        self.sfx = [None]
        self.trn = [[]]
        self.rsfx= [[]]
        self.lrs = [0]
        self.symbol= [0] 

        # The oracle's initial state, hiding the actual implementation
        self.initial_state = 0
                        
        # Compression attributes
        self.compror = []
        self.code = []    
        self.seg = []   
        
        # Object attributes
        self.kind = 'f'
        self.name = ''
        
        # Oracle statistics
        self.n_states = 1
        self.max_lrs = [0]
        self.avg_lrs = [0.]

        # Reachability class via (possibly reverse) suffix links
        self.suffix_class = [set([0])]

    def update_params(self, **kwargs):
        """Subclass this"""
        self.params.update(kwargs)
        
    def add_state(self, new_symbol):
        # TODO : finish this
        """Skeleton of a generic add_state function. 

        Not being called by any of the implementations for now.
        """ 
        def _init_new_state(self):
            """Initialize content holders for the newly-added state.

            Return the index of the new state.
            """
            self.sfx.append(0)
            self.rsfx.append([])
            self.trn.append([])
            self.lrs.append(0)
            self.symbol.append(new_symbol)
        
            self.n_states += 1
        
            new_i = self.n_states - 1  # Index of the newly added state

            self.trn[new_i - 1].append(new_i)
            _init_sfx = self.sfx[new_i - 1]
            _init_pi_1 = new_i - 1

            # `new_i` is reachable from `new_i` by an empty path
            self.suffix_class.append(set(new_i))
            
            return new_i, _init_sfx, _init_pi_1

        new_i, current_suffix, pi_1 = self._init_new_state()

        while current_suffix != None:  # Follow chain of suffixes
            # Check whether the current state holds a transition labeled
            # by new_symbol
            can_read = self.read_symbol(current_suffix, new_symbol) 
            if not can_read:
                self.trn[current_suffix].append(new_i)
                pi_1 = current_suffix
                current_suffix = self.sfx[current_suffix]
            else:
                break
                
        if current_suffix == None:
            self.sfx[new_i] = 0
            self.lrs[new_i] = 0
        else:
            self.sfx[new_i] = self.read_symbol(current_suffix, new_symbol)
            self.lrs[new_i] = 1 + self._len_common_suffix(pi_1,
                                                          self.sfx[new_i] - 1)

        previous_suffix_pos = new_i - self.lrs[new_i]
        current_suffix = self._find_better(new_i,
                                           self.symbol[previous_suffix_pos])

        if current_suffix is not None:
            self.lrs[new_i] += 1
            self.sfx[i] = current_suffix

        self.rsfx[self.sfx[new_i]].append(new_i)
        
        """Subclass this"""
        pass

    def update_suffix_class(self, state):
        """Update `state`'s suffix reachability class."""
        suffix = self.sfx[state]
        # Init new reachability class
        self.suffix_class.append(set([state]))
        # Add `suffix`'s reachable states to `state`'s reachability class
        self.suffix_class[state].update(self.suffix_class[suffix])
    
        if self.include_rsfx():
            # Allow moves through reverse suffix links on the oracle
            # Add `new_i` to the chosen suffix's class and all its members
            # `copy()` is required, because `suffix`'s suffix-class will
            # grow too.

            # Note : a special case is required for state `0`, because
            # allowing reverse suffix moves from `0` nullifies all suffix
            # structure (it makes the suffix links graph strongly connected)
            for s in self.suffix_class[suffix].copy():
                if s != 0:
                    self.suffix_class[s].add(state)
            
    
    def labeled_links_state(self, state):
        """List the states reachable by a compressed labeled link from `state`.

        Compressed labeled links are either a forward link or
        a sequence of (possibly reverse) suffix links and a final forward link.

        Keyword arguments:
            state: int
                The state for which to extract the compressed labeled links
        """
        labeled = set()
        if state == self.initial_state or self.sfx[state] == self.initial_state:
            labeled.update(*[self.trn[s] for s in self.suffix_class[state]])
        else:
            # Don't allow not previously existing jumps from the initial state:
            # this connects all states and nullifies the whole structure
            # of the oracle
            labeled.update(*[self.trn[s] for s in self.suffix_class[state]
                             if s != self.initial_state])
        return labeled

    
    def read_symbol_forward(self, state, symbol):
        """Return the state reached by reading <symbol> in <state>.

        Return None if the given symbol is not recognized in <state>.

        Keyword arguments:
            state: int
                A state in the oracle.
            symbol: an oracle symbol (implementation dependant)
                The symbol to read.
        """
        _same_symbol_neighbour = [goal for goal in self.trn[state]
                                  if self.symbol[goal] == symbol]
        if len(_same_symbol_neighbours) > 0:
            # len(_same_symbol_neighbour) == 1 since the oracle is deterministic
            return _same_symbol_neighbour[0]
        else:
            return None

    def read_feature(self, state, feature):
        """Return all neighbours from <state> with a feature close to <feature>.

        The returned states are sorted according to their distance to <feature>.
        """
        neighbour_features = [self.feature[goal] for goal in self.trn[state]]
        dists = distances.cdist([feature], neighbour_features,
                                dfunc=self.params['dfunc'])[0]
        dists_decorated = [(self.trn[state][i], dists[i])
                           for i in range(len(dists))
                           if dists[i] < self.params['threshold']]
        dists_sorted = sorted(dists_decorated,
                              key=lambda state_dist: state_dist[1])
        return dists_sorted
        
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
                self.symbol[j-self.lrs[i]] == symbol):
                return j
        return None
                    
    @property
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
        """Non-overlapping Compror"""
        if (self.seg == []):
            j = 0
        else:
            j = self.seg[-1][1]
            last_len = self.seg[-1][0]
            if last_len + j > self.n_states:
                return

        i = j
        while (j < self.n_states-1):
            while (i < self.n_states - 1 and
                   self.lrs[i + 1] >= i - j + 1):
                i = i + 1
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
            h1[i] = vutl.entropy(cw[0:i + 1])

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
        """Dispatch between different types of information-rate computation"""
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
        threshold = self.params.get('threshold')
        if (threshold is not None):
            return int(threshold)  # Todo, no need for int here
        else:
            raise ValueError("Threshold is not set!")

    def dfunc(self):
        dfunc = self.params.get('dfunc')
        if (dfunc is not None):
            return dfunc
        else:
            raise ValueError("dfunc is not set!")
    
    def dfunc_handle(self, a, b_vec):     
        fun = self.params['dfunc_handle']  
        return fun(a, b_vec)

    def include_rsfx(self):
        include_rsfx = self.params.get('include_rsfx')
        if (include_rsfx is not None):
            return include_rsfx
        else:
            raise ValueError("include_rsfx is not set!")

    
class FO(FactorOracle):
    """An implementation of the factor oracle."""
    def __init__(self, **kwargs):
        super(FO, self).__init__(**kwargs)
        self.reset(**kwargs)
        
    def reset(self, **kwargs):
        super(FO, self).reset(**kwargs)
        self.kind = 'r'

    def add_state(self, new_symbol):
        """

        :type self: oracle
        """
        # Initialize data structures for the state
        self.sfx.append(0)
        self.rsfx.append([])
        self.trn.append([])
        self.lrs.append(0)
        self.symbol.append(new_symbol)
        
        self.n_states += 1
        
        new_i = self.n_states - 1  # Index of the newly added state
        
        self.trn[new_i - 1].append(new_i)
        current_suffix = self.sfx[new_i - 1]
        pi_1 = new_i-1
        
        # Add forward links
        while current_suffix != None:
            _symbols = [self.symbol[state] for state in self.trn[current_suffix]]    
            if self.symbol[new_i] not in _symbols:
                self.trn[current_suffix].append(new_i)
                pi_1 = current_suffix
                current_suffix = self.sfx[current_suffix]
            else:
                break
        
        if current_suffix == None:
            # No repeated suffix found, link to initial state
            self.sfx[new_i] = 0
            self.lrs[new_i] = 0
        else:
            # A repeated suffix was found
            _out_trn_sharing_symbol = [state for state in self.trn[current_suffix]
                                       if self.symbol[state] == new_symbol]
            _out_trn_sharing_symbol = sorted(_out_trn_sharing_symbol)
            _first_state_sharing_symbol = _out_trn_sharing_symbol[0]
            self.sfx[new_i] = _first_state_sharing_symbol
            self.lrs[new_i] = self._len_common_suffix(pi_1, self.sfx[new_i]-1) + 1
        
        current_suffix = self._find_better(new_i,
                                           self.symbol[new_i-self.lrs[new_i]])
        if current_suffix != None:
            self.lrs[new_i] += 1
            self.sfx[new_i] = current_suffix
        self.rsfx[self.sfx[new_i]].append(new_i)

        new_max_lrs = max(self.lrs[new_i], self.max_lrs[new_i-1])
        self.max_lrs.append(new_max_lrs)
        
        N = float(self.n_states)
        previous_average_lrs = self.avg_lrs[new_i-1]
        self.avg_lrs.append(previous_average_lrs*((N-1)/N) + 
                            self.lrs[new_i]*(1/N))

        self.update_suffix_class(new_i)
        
    def accept(self, context):
        """Check if the context could be accepted by the oracle
        
        Keyword arguments:
            context: oracle's symbols sequence
                sequence same type as the oracle symbols
        Returns:
            bAccepted: whether the sequence is accepted (resp. rejected)
            _next: the state where the sequence is accepted (resp. rejected)
        """
        _next = 0
        for _s in context:
            _symbols = [self.symbol[j] for j in self.trn[_next]]
            if _s in _symbols:
                _next = self.trn[_next][_symbols.index(_s)]
            else:
                return 0, _next
        return 1, _next

    def get_alphabet(self):
        alphabet = [self.symbol[i] for i in self.trn[0]]
        dictionary = dict(zip(alphabet, range(len(alphabet))))
        return dictionary


class MO(FactorOracle):
    def __init__(self, **kwargs):
        super(MO, self).__init__(**kwargs)
        self.reset(**kwargs)
        
    def reset(self, suffix_method='inc', **kwargs):
        super(MO, self).reset(**kwargs)

        # Reset class specific attributes
        self.kind = 'a'
        # self.f_array = [0]
        self.feature = feature_array(self.params['dim'])
        self.feature.add(np.zeros(self.params['dim'], ))
       
        self.feature = [0]
        self.symbol[0] = 0
        self.latent = []
        self.suffix_method = suffix_method
        
    def add_state(self, new_feature):
        """Create new state and update related links and compressed state"""
        self.sfx.append(0)
        self.rsfx.append([])
        self.trn.append([])
        self.lrs.append(0)

        self.feature.add(new_feature)  # Experiment using pointers

        self.n_states += 1
        new_i = self.n_states - 1  # Index of the newly added state

        # assign new transition from state new_i-1 to new_i
        self.trn[new_i - 1].append(new_i)
                
        """Experiment enforcing continuity"""
#        if new_i != 1:
#            self.trn[new_i - 1].append(new_i-1)
#            k = new_i - 1
#        else:
#            k = self.sfx[new_i - 1]

        current_suffix = self.sfx[new_i - 1]
        pi_1 = new_i - 1
    
        # Iteratively backtrack through suffixes starting from state new_i-1
        dvec = []
        if self.suffix_method == 'inc':
            suffix_candidate = 0
        elif self.suffix_method == 'complete':
            suffix_candidate = []
        else:
            suffix_candidate = 0
        
        while current_suffix is not None:
            close_features = self.read_feature(current_suffix, new_feature)

            if len(close_features) == 0:  # If no transition from suffix
                # Add new forward link to unvisited state
                self.trn[current_suffix].append(new_i)
                
                pi_1 = current_suffix
                if self.suffix_method != 'complete':
                    current_suffix = self.sfx[current_suffix]
            else:
                closest_feature, lowest_distance = close_features[0]
                if self.suffix_method == 'inc':
                    suffix_candidate = closest_feature
                    break
                elif self.suffix_method == 'complete':
                    suffix_candidate.append(closest_feature, lowest_distance)
                else:
                    suffix_candidate = closest_feature
                    break
            
            if self.suffix_method == 'complete':
                current_suffix = self.sfx[current_suffix]

        if self.suffix_method == 'complete':
            if suffix_candidate == []:
                self.sfx[new_i] = 0
                self.lrs[new_i] = 0
                self.latent.append([new_i])
                self.symbol.append(len(self.latent)-1)
            else:
                sorted_suffix_candidates = sorted(suffix_candidate,
                                                  key=lambda suffix:suffix[1])
                self.sfx[new_i] = sorted_suffix_candidates[0][0]
                lrs_new_i = self._len_common_suffix(pi_1, self.sfx[new_i]-1) + 1
                self.lrs[new_i] = lrs_new_i
                self.latent[self.symbol[self.sfx[new_i]]].append(new_i)
                self.symbol.append(self.symbol[self.sfx[new_i]])
        else:
            if current_suffix == None:
                self.sfx[new_i] = 0
                self.lrs[new_i] = 0
                self.latent.append([new_i])
                self.symbol.append(len(self.latent)-1)
            else:
                self.sfx[new_i] = suffix_candidate
                lrs_new_i = self._len_common_suffix(pi_1, self.sfx[new_i]-1) + 1
                self.lrs[new_i] = lrs_new_i
                self.latent[self.symbol[self.sfx[new_i]]].append(new_i)
                self.symbol.append(self.symbol[self.sfx[new_i]])
            
        self.rsfx[self.sfx[new_i]].append(new_i)
        
        if self.lrs[new_i] > self.max_lrs[new_i-1]:
            self.max_lrs.append(self.lrs[new_i])
        else:
            self.max_lrs.append(self.max_lrs[new_i-1])

        N = float(self.n_states)
        previous_average_lrs = self.avg_lrs[new_i-1]
        self.avg_lrs.append(previous_average_lrs*(N-1)/N + 
                            self.lrs[new_i]*(1/N))
    
        self.update_suffix_class(new_i)
        

class VMO(FactorOracle):
    def __init__(self, **kwargs):
        super(VMO, self).__init__(**kwargs)
        self.reset(**kwargs)
     
    def reset(self, **kwargs):
        super(VMO, self).reset(**kwargs)

        # Reset class-specific attributes
        self.kind = 'v'
        self.feature = [0]
        self.symbol[0] = 0
        self.latent = []
        self.centroid = []
        self.hist = []
        
        # Connectivity-related attributes
        self.con = []     
        self.transition = []

    def add_state(self, new_feature):
        self.sfx.append(0)
        self.rsfx.append([])
        self.trn.append([])
        self.lrs.append(0)

        self.feature.append(new_feature)  # Experiment using pointers
        
        self.n_states += 1
        new_i = self.n_states - 1  # Index of the newly added state
    
        # Assign new transition from state new_i-1 to new_i
        self.trn[new_i-1].append(new_i)
        k = self.sfx[new_i - 1]
        pi_1 = new_i - 1
        
        dvec = []
        trn_list = []
        suffix_candidate = 0
        
        while k is not None:
            suffix_features = [self.centroid[self.symbol[t]] for t in self.trn[k]]
            dvec = distances.cdist([new_feature], suffix_features,
                                   dfunc=self.params['dfunc'])[0]
            close_features = np.where(dvec < self.params['threshold'])

            if len(close_features) == 0:  # If no transition from suffix
                # Create a new forward link to unvisited state
                self.trn[k].append(new_i)
                trn_list.append(k)
                pi_1 = k
                k = self.sfx[k]
            else:
                earliest_close_feat = close_features[np.argmin(
                    dvec[close_features])]
                suffix_candidate = self.trn[k][earliest_close_feat]
                trn_list.append(new_i-1)
                break

        if k == None:
            # Beginning of the input sequence reached, no repeated suffix found
            self.sfx[new_i] = 0
            self.lrs[new_i] = 0
            self.latent.append([new_i])
            self.hist.append(1)
            self.symbol.append(len(self.latent)-1)
            self.centroid.append(new_feature)
            if new_i > 1:
                self.con[self.symbol[new_i-1]].add(self.symbol[new_i])
                self.con.append(set([self.symbol[new_i]]))
            else:
                self.con.append({})
        else:
            # k is the first state of the longest repeated suffix
            self.sfx[new_i] = suffix_candidate
            self.lrs[new_i] = self._len_common_suffix(pi_1, self.sfx[new_i]-1) + 1
            _suffix_symbol = self.symbol[self.sfx[new_i]]
            self.latent[_suffix_symbol].append(new_i)
            self.hist[_suffix_symbol] += 1
            self.symbol.append(_suffix_symbol)
            self.centroid[_suffix_symbol] = ((self.centroid[_suffix_symbol] *
                                              (self.hist[_suffix_symbol]-1)
                                              + new_feature) /
                                              self.hist[_suffix_symbol])
            self.con[self.symbol[new_i-1]].add(self.symbol[new_i])
            map(set.add, [self.con[self.symbol[c]] for c in trn_list],
                [self.symbol[new_i]]*len(trn_list))
        self.rsfx[self.sfx[new_i]].append(new_i)
        
        if self.lrs[new_i] > self.max_lrs[new_i-1]:
            self.max_lrs.append(self.lrs[new_i])
        else:
            self.max_lrs.append(self.max_lrs[new_i-1])

        N = float(self.n_states)
        previous_average_lrs = self.avg_lrs[new_i-1]
        self.avg_lrs.append(previous_average_lrs*((N-1)/N) +
                            self.lrs[new_i]*(1/N))

        self.update_suffix_class(new_i)

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
    """Create a factor oracle based on the input type."""
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

def _build_oracle(flag, oracle, input_features):
    if (type(input_features) != np.ndarray or
        type(input_features[0]) != np.ndarray):
        input_features = np.array(input_features)

    if input_features.ndim != 2:
        input_features = np.expand_dims(input_features, axis=1)
    
    for obs in input_features:
        oracle.add_state(obs)

    if flag is 'a':
        oracle.feature.finalize()
        
    return oracle
 
def build_oracle(input_features, flag, threshold=0, suffix_method='inc', 
                 feature=None, weights=None, dfunc='cosine',
                 dfunc_handle=None, dim=1):
    
    # Initialize `weights` if needed
    if weights is None:
        weights = {}
        weights.setdefault(feature, 1.0)

    if flag in ['a', 'f', 'v']:
        oracle = _create_oracle(flag, threshold=threshold, dfunc=dfunc,
                                dfunc_handle=dfunc_handle, dim=dim)
        oracle = _build_oracle(flag, oracle, input_features)
    else:
        oracle = _create_oracle('a', threshold=threshold, dfunc=dfunc,
                                dfunc_handle=dfunc_handle, dim=dim)
        oracle = _build_oracle(flag, oracle, input_features, suffix_method)
    
    return oracle

def find_threshold_sgd(input_features, r=(0, 1, 0.1), method='ir', flag='a',
                       suffix_method='inc', alpha=1.0, feature=None, ir_type='cum',
                       dfunc='cosine', dfunc_handle=None, dim=1):
    thresholds = np.arange(r[0], r[1], r[2])
    prev_ir = 0
    for t in thresholds:
        tmp_oracle = build_oracle(input_features, flag=flag, threshold=t,
                                  suffix_method=suffix_method, feature=feature,
                                  dfunc=dfunc, dfunc_handle=dfunc_handle, dim=dim)
        tmp_ir, h0, h1 = tmp_oracle.IR(ir_type=ir_type, alpha=alpha)
        ir = tmp_ir.sum()
        if ir < prev_ir:
            return t - r[2], prev_ir
        else:
            prev_ir = ir


def find_threshold_nt(input_features, r=(0, 1, 0.1), method='ir', flag='a',
                      suffix_method='inc', alpha=1.0, feature=None, ir_type='cum',
                      dfunc='cosine', dfunc_handle=None, dim=1):
    t = r[1] / 2.0
    tmp_oracle = build_oracle(input_features, flag=flag, threshold=t,
                              suffix_method=suffix_method, feature=feature,
                              dfunc=dfunc, dfunc_handle=dfunc_handle, dim=dim)
    tmp_ir, h0, h1 = tmp_oracle.IR(ir_type=ir_type, alpha=alpha)
    ir = tmp_ir.sum()

    t_tmp = t + r[2]
    tmp_oracle = build_oracle(input_features, flag=flag, threshold=t_tmp,
                              suffix_method=suffix_method, feature=feature,
                              dfunc=dfunc, dfunc_handle=dfunc_handle, dim=dim)
    tmp_ir, h0, h1 = tmp_oracle.IR(ir_type=ir_type, alpha=alpha)
    ir_tmp = tmp_ir.sum()

    if ir_tmp > ir:
        while ir_tmp > ir:
            t = t_tmp
            ir = ir_tmp

            t_tmp = t + r[2]
            tmp_oracle = build_oracle(input_features, flag=flag, threshold=t_tmp,
                                      suffix_method=suffix_method, feature=feature,
                                      dfunc=dfunc, dfunc_handle=dfunc_handle, dim=dim)
            tmp_ir, h0, h1 = tmp_oracle.IR(ir_type=ir_type, alpha=alpha)
            ir_tmp = tmp_ir.sum()
    else:
        t_tmp = t - r[2]
        tmp_oracle = build_oracle(input_features, flag=flag, threshold=t_tmp,
                                  suffix_method=suffix_method, feature=feature,
                                  dfunc=dfunc, dfunc_handle=dfunc_handle, dim=dim)
        tmp_ir, h0, h1 = tmp_oracle.IR(ir_type=ir_type, alpha=alpha)
        ir_tmp = tmp_ir.sum()

        while ir_tmp > ir:
            t = t_tmp
            ir = ir_tmp

            t_tmp = t - r[2]
            tmp_oracle = build_oracle(input_features, flag=flag, threshold=t_tmp,
                                      suffix_method=suffix_method, feature=feature,
                                      dfunc=dfunc, dfunc_handle=dfunc_handle, dim=dim)
            tmp_ir, h0, h1 = tmp_oracle.IR(ir_type=ir_type, alpha=alpha)
            ir_tmp = tmp_ir.sum()
    return t, ir


def find_threshold(input_features, r=(0, 1, 0.1), method='ir', flag='a',
                   suffix_method='inc', alpha=1.0, feature=None, ir_type='cum',
                   dfunc='cosine', dfunc_handle=None, dim=1,
                   verbose=False, entropy=False):
    if method == 'ir':
        return find_threshold_ir(input_features, r, flag, suffix_method, alpha,
                                 feature, ir_type, dfunc, dfunc_handle, dim,
                                 verbose, entropy)
    # elif method == 'motif':
    #     return find_threshold_motif(input_features, r, flag, suffix_method, alpha,
    #                                 feature, dfunc, dfunc_handle, dim, verbose)


def find_threshold_ir(input_features, r=(0,1,0.1), flag='a', suffix_method='inc',
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
            print('Testing threshold: ' + str(t) + '\n')
        tmp_oracle = build_oracle(input_features, flag=flag, threshold=t,
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


# def find_threshold_motif(input_features, r=(0, 1, 0.1), flag='a',
#                          suffix_method='inc', alpha=1.0, feature=None,
#                          dfunc='euclidean', dfunc_handle=None, dim=1,
#                          verbose=False):
#     thresholds = np.arange(r[0], r[1], r[2])
#     avg_len = []
#     avg_occ = []
#     avg_num = []
#
#     for t in thresholds:
#         tmp_oracle = build_oracle(input_features, flag=flag, threshold=t,
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
