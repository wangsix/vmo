'''
oracle.py
Variable Markov Oracle in python

Copyright (C) 7.28.2013 Cheng-i Wang, Greg Surges

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
'''

import numpy as np
from itertools import izip
from matplotlib.mlab import find


class data(object):
    """A helper class for construct data object for symbolic comparison
    
    By default, the fist entry of the list or tuple is used as the feature to 
    test for equality between different data object. 
    
    Attributes:
        content: a list or tuple
        idx: the index of the list or tuple to be tested for equality 
    """
    def __init__(self, data_item, index = 0):
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
        if type(other) == data:
            if self.content[self.idx] == other.content[self.idx]:
                return False
            else:
                return True
        else:
            return True

class FactorOracle(object):
    """ The base class for the FO(factor oracle) and MO(variable markov oracle)
    
    Attributes:
        sfx: a list containing the suffix link of each state.
        trn: a list containing the forward links of each state as a list.
        rsfx: a list containing the reverse suffix links of each state 
            as a list.
        lrs: the value of longest repeated suffix of each state.
        data: the object/value associated with the direct link 
            connected to each state.
        compror: a list of tuples (i, i-j), i is the current coded position,
            i-j is the length of the corresponding coded words.
        code: a list of tuples (len, pos), len is the length of the 
            corresponding coded words, pos is the position where the coded
            words starts.
        seg: same as code but non-overlapping.
        kind: 
            'a': audio oracle
            'f': repeat oracle
        n_states: number of total states, also is length of the input 
            sequence plus 1.
        max_lrs: the longest lrs so far.
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
        self.rsfx= []
        self.lrs = []
        self.data= [] 
        
        # Compression attributes
        self.compror = []
        self.code = []    
        self.seg = []   
        
        # Object attributes
        self.kind = 'f'
        self.name = ''
        
        # Oracle statistics
        self.n_states = 1
        self.max_lrs = 0
        
        # Oracle parameters
        self.params = {
                       'threshold':0,
                       'weights': {},
                       'dfunc': 'euclidean',
                       'dfunc_handle':None
                       }
        self.update_params(**kwargs)
        
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

    def encode(self): #Referenced from IR module
        if self.compror == []:
            j = 0
        else:
            j = self.compror[-1][0]
            
        i = j
        while j < self.n_states-1:
            while i < self.n_states - 1 and self.lrs[i + 1] >= i - j + 1:
                i = i + 1
            if i == j:
                i = i + 1
                self.code.append((0,i))
                self.compror.append((i,0))
            else:
                self.code.append((i - j, self.sfx[i] - i + j + 1))
                self.compror.append((i,i-j)) 
            j = i
        return self.code, self.compror
        
    def segment(self):
        """An non-overlap version Compror"""
        if self.seg == []:
            j = 0
        else:
            j = self.seg[-1][0]

        i = j
        while j < self.n_states-1:
            while i < self.n_states - 1 and self.lrs[i + 1] >= i - j + 1:
                i = i + 1
            if i == j:
                i = i + 1
                self.seg.append((0, i))
            else:
                if (self.sfx[i] + self.lrs[i]) <= i:
                    self.seg.append((i - j, self.sfx[i] - i + j + 1))

                else:
                    _i = j + i - self.sfx[i]
                    self.seg.append((_i-j ,self.sfx[i] - i + j + 1))
                    _j = _i
                    while _i < i and self.lrs[_i + 1] - self.lrs[_j]  >= _i - _j + 1:
                        _i = _i +1
                    if _i == _j:
                        _i = _i + 1
                        self.seg.append((0, _i))
                    else:
                        self.seg.append((_i-_j, self.sfx[_i]-_i+_j+1))
            j = i
        return self.seg
    
    def _ir(self, alpha = 1.0):
        """Referenced from IR.py
        """
        code, _ = self.encode()
        cw = np.zeros(len(code))
        for i, c in enumerate(code):
            cw[i] = c[0]+1
    
        c0 = [1 if x[0] == 0 else 0 for x in self.code]
        h0 = np.log2(np.cumsum(c0))
#         h0 = np.array([np.log2(x) for x in np.cumsum(c0)])
    
        dti = [1 if x[0] == 0 else x[0] for x in self.code]
        ti = np.cumsum(dti)
    
        h = np.zeros(len(cw))
    
        for i in range(1, len(cw)):
            h[i] = _entropy(cw[0:i+1])
    
#         h = np.array(h)
#         h0 = np.array(h0)
        ir = ti, alpha*h0-h
    
        return ir, self.code, self.compror
            
    def _ir_cum(self, alpha=1.0):
        code, _ = self.encode()
        
        N = self.n_states
    
        cw0 = np.zeros(N) #cw0 counts the appearance of new states only 
        cw1 = np.zeros(N) #cw1 counts the appearance of all compror states
        BL = np.zeros(N)  #BL is the block length of compror codewords
    
        j = 0
        for i in range(len(code)):
            if self.code[i][0] == 0:
                cw0[j] = 1
                cw1[j] = 1
                BL[j] = 1
                j = j+1
            else:
                L = code[i][0]    
#                 cw0[j:j+L] = np.zeros(L)
                cw1[j] = 1
#                 cw1[j:j+L] = np.concatenate(([1], np.zeros(L-1)))
                BL[j:j+L] = L #range(1,L+1)
                j = j+L
    
        H0 = np.log2(np.cumsum(cw0))
        H1 = np.log2(np.cumsum(cw1))
        H1 = H1/BL
        ir = alpha*H0 - H1
        ir[ir<0] = 0
#         ir = np.max(np.append(ir,0))
        
        return ir, self.code, self.compror        
    
    def IR(self, alpha = 1.0, ir_type = 'cum'):
        if ir_type == 'cum':
            ir, code, compror = self._ir_cum(alpha)
        else:
            ir, code, compror = self._ir(alpha)
        return ir, code, compror 
        
    def num_clusters(self):
        return len(self.rsfx[0])
    
    def threshold(self):
        if self.params.get('threshold'):
            return int(self.params.get('threshold'))
        else:
            raise ValueError("Threshold is not set!")
        
    def weights(self):
        if self.params.get('weights'):
            return self.params.get('weights')
        else:
            raise ValueError("Weights are not set!")
        
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
#             while self.sfx[p1] != self.sfx[p2]:
                p2 = self.sfx[p2]
        return min(self.lrs[p1], self.lrs[p2])
    
    def _find_better(self, i, symbol):
        self.rsfx[i].sort()
        for j in self.rsfx[i]:
            if self.lrs[j] == self.lrs[i] and self.data[j-self.lrs[i]] == symbol:
                return j
        return None   
    
class FO(FactorOracle):
    """ An implementation of the factor oracle
    """
    
    def __init__(self, **kwargs):
        super(FO, self).__init__(**kwargs)
        self.kind = 'r'
        
    def add_state(self, new_symbol):
        self.sfx.append(0)
        self.rsfx.append([])
        self.trn.append([])
        self.lrs.append(0)
        self.data.append(new_symbol)
        
        self.n_states += 1
        
        i = self.n_states - 1
        
        self.trn[i-1].append(i)
        k = self.sfx[i-1]
        pi_1 = i-1
        
        # Adding forward links
        while k != None:
            _symbols = [self.data[state] for state in self.trn[k]]    
            if self.data[i] not in _symbols:
                self.trn[k].append(i)
                pi_1 = k
                k = self.sfx[k]
            else:
                break
        
        if k == None:
            self.sfx[i] = 0
            self.lrs[i] = 0
        else:
            _query = [[self.data[state], state] for state in self.trn[k] if self.data[state] == self.data[i]]
            _query = sorted(_query, key=lambda _query: _query[1])
            _state = _query[0][1]  
            self.sfx[i] = _state
            self.lrs[i] = self._len_common_suffix(pi_1, self.sfx[i]-1) + 1
        
        k = self._find_better(i, self.data[i-self.lrs[i]])
        if k != None:
            self.lrs[i] += 1
            self.sfx[i] = k
        self.rsfx[self.sfx[i]].append(i)
        
        if self.lrs[i] > self.max_lrs:
            self.max_lrs = self.lrs[i]
    
    def accept(self, context):
        """ Check if the context could be accepted by the oracle
        
        Args:
            context: s sequenc same type as the oracle data
        
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
        self.feature = [0]
        self.f_array = [0]
        self.data[0] = None
        self.latent = []
        
        # including connectivity
        self.con = []
    
    def add_state(self, new_data):
        """Create new state and update related links and compressed state"""
        self.sfx.append(0)
        self.rsfx.append([])
        self.trn.append([])
        self.lrs.append(0)

        # Experiment with pointer-based  
        self.feature.append(new_data)
        self.f_array.append(new_data)        

        self.n_states += 1 
        i = self.n_states - 1
    
        # assign new transition from state i-1 to i
        self.trn[i - 1].append(i)
        k = self.sfx[i - 1] 
        pi_1 = i - 1
    
        # iteratively backtrack suffixes from state i-1
        dvec = []
        trn_list = []
        suffix_candidate = 0        
        '''
        c = list(itertools.chain.from_iterable([self.latent[_c] for _c in list(self.con[self.data[k]])]))        
        if self.params['dfunc'] == 'euclidean':
            a = np.array(f) - np.array([self.f_array[t] for t in c])
            dvec = np.sqrt((a*a).sum(axis=1)) 
        I = find(dvec < self.params['threshold'])
        '''
        
        while k != None:
            ''' OLD implementation
            dvec = [get_distance(new_data, self.feature[s], self.params['weights'], self.params['dfunc']) for s in self.trn[k] if s != 0]
            dvec = array(dvec)
            '''
            # NEW Implementation
            if self.params['dfunc'] == 'euclidean':
                a = np.array(new_data) - np.array([self.f_array[t] for t in self.trn[k]])
                dvec = np.sqrt((a*a).sum(axis=1))
            elif self.params['dfunc'] == 'other':
                dvec = self.dfunc_handle(new_data, [self.f_array[t] for t in self.trn[k]])
                
            # if no transition from suffix
            I = find(dvec < self.params['threshold'])
            if len(I) == 0:
                self.trn[k].append(i) # Create a new forward lint to unvisited state
                trn_list.append(k)
                pi_1 = k
                k = self.sfx[k]
            else:
                suffix_candidate = self.trn[k][I[np.argmin(dvec[I])]]
                trn_list.append(i-1)
                break
#             k = self.sfx[k]

        if k == None:
            self.sfx[i] = 0
            self.lrs[i] = 0
            self.latent.append([i])
            self.data.append(len(self.latent)-1)
            if i > 1:
                self.con[self.data[i-1]].add(self.data[i]) 
                self.con.append(set([self.data[i]]))
            else:
                self.con.append(set([]))
        else:
            self.sfx[i] = suffix_candidate
            self.lrs[i] = self._len_common_suffix(pi_1, self.sfx[i]-1) + 1
            self.latent[self.data[self.sfx[i]]].append(i)
            self.data.append(self.data[self.sfx[i]])
            self.con[self.data[i-1]].add(self.data[i])
            map(set.add, [self.con[self.data[c]] for c in trn_list], [self.data[i]]*len(trn_list))
        self.rsfx[self.sfx[i]].append(i)
        
        if self.lrs[i] > self.max_lrs:
            self.max_lrs = self.lrs[i]
            
                         
def _entropy(x):
    x = np.divide(x, sum(x), dtype = float)
    return sum(np.multiply(-np.log2(x),x))

def _create_oracle(oracle_type, **kwargs):
    """A routine for creating a factor oracle."""
    if oracle_type == 'f':
        return FO(**kwargs)
    elif oracle_type == 'a':
        return MO(**kwargs)
    else:
        return MO(**kwargs)

def _build_factor_oracle(oracle, input_data):    
    for obs in input_data:
        oracle.add_state(obs)
    return oracle
 
def build_oracle(input_data, flag, threshold = 0, feature = None, weights = None, dfunc = 'euclidean', dfunc_handle = None):
    
    # initialize weights if needed 
    if weights == None:
        weights = {}
        weights.setdefault(feature, 1.0)

    if flag == 'a' or flag == 'f':
        oracle = _create_oracle(flag, threshold = threshold, weights = weights, dfunc = dfunc, dfunc_handle = dfunc_handle)
    else:
        oracle = _create_oracle('a', threshold = threshold, weights = weights, dfunc = dfunc, dfunc_handle = dfunc_handle)
             
    oracle = _build_factor_oracle(oracle, input_data)
    return oracle 
    

def find_threshold(input_data, r = (0,1,0.1), flag = 'a', feature = None, ir_type='cum', dfunc ='euclidean', dfunc_handle = None, VERBOSE = False):
    thresholds = np.arange(r[0], r[1], r[2])
    irs = []
    for t in thresholds:
        if VERBOSE:
            print 'testing threshold:', t
        tmp_oracle = build_oracle(input_data, flag = flag, threshold = t, feature = feature, dfunc = dfunc, dfunc_handle = dfunc_handle)
        tmp_ir, code, compror = tmp_oracle.IR(ir_type = ir_type)
        # is it a sum?
        if ir_type=='old' or ir_type=='cum':
            irs.append(tmp_ir.sum())
        else:
            irs.append(tmp_ir[1].sum())
    # now pair irs and thresholds in a vector, and sort by ir
    ir_thresh_pairs = [(a,b) for a, b in zip(irs, thresholds)]
    pairs_return = ir_thresh_pairs
    ir_thresh_pairs = sorted(ir_thresh_pairs, key= lambda x: x[0], reverse = True)
    return ir_thresh_pairs[0], pairs_return     
    
    