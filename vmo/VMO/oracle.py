'''
oracle.py
Variable Markov Oracle in python

Copyright (C) 9.2014 Cheng-i Wang, Greg Surges

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
        self.max_lrs = []
        self.max_lrs.append(0)
        self.avg_lrs = []
        self.avg_lrs.append(0.0)
        
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
    
    def reset(self, **kwargs):
        self.update_params(**kwargs)
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
                _code.append((0,i))
                _compror.append((i,0))
            else:
                _code.append((i - j, self.sfx[i] - i + j + 1))
                _compror.append((i,i-j)) 
            j = i
        return _code, _compror
        
    def encode(self): #Referenced from IR module
        _c, _cmpr = self._encode()
        self.code.extend(_c)
        self.compror.extend(_cmpr)
        """
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
        """
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
        cw = np.zeros(len(code)) # Number of code words
        for i, c in enumerate(code):
            cw[i] = c[0]+1
    
        c0 = [1 if x[0] == 0 else 0 for x in self.code]
        h0 = np.log2(np.cumsum(c0))
    
        dti = [1 if x[0] == 0 else x[0] for x in self.code]
        ti = np.cumsum(dti)
    
        h = np.zeros(len(cw))
    
        for i in range(1, len(cw)):
            h[i] = _entropy(cw[0:i+1])
    
        ir = ti, alpha*h0-h
    
        return ir, self.code, self.compror
    
    def _ir_fixed(self):
        code, _ = self.encode()
         
#         if self.kind == 'v':
#             p = np.array([len(sym) for sym in self.latent])
#             p = p/(self.n_states-1.0)
#             h0 = np.log2(p).dot(-p)
#         else:
#             h0 = np.log2(self.num_clusters())

        h0 = np.log2(self.num_clusters())
        
        if self.max_lrs[-1] == 0:
            h1 = np.log2(self.n_states-1) 
        else:
            h1 = np.log2(self.n_states-1) + np.log2(self.max_lrs[-1])
                
        BL = np.zeros(self.n_states-1)        
        j = 0
        for i in range(len(code)):
            if self.code[i][0] == 0:
                BL[j] = 1
                j = j+1
            else:
                L = code[i][0]    
                BL[j:j+L] = L #range(1,L+1)
                j = j+L
        ir = h0 - h1/BL
        ir[ir<0] = 0
        return ir, self.code, self.compror
        
    def _ir_cum(self, alpha=1.0):
        code, _ = self.encode()
        
        N = self.n_states
    
        cw0 = np.zeros(N-1) #cw0 counts the appearance of new states only 
        cw1 = np.zeros(N-1) #cw1 counts the appearance of all compror states
        BL = np.zeros(N-1)  #BL is the block length of compror codewords
    
        j = 0
        for i in range(len(code)):
            if self.code[i][0] == 0:
                cw0[j] = 1
                cw1[j] = 1
                BL[j] = 1
                j = j+1
            else:
                L = code[i][0]    
                cw1[j] = 1
                BL[j:j+L] = L #range(1,L+1)
                j = j+L
    
        H0 = np.log2(np.cumsum(cw0))
        H1 = np.log2(np.cumsum(cw1))
        H1 = H1/BL
        ir = alpha*H0 - H1
        ir[ir<0] = 0
        
        return ir, self.code, self.compror 
        
    def _ir_cum2(self):
        code, _ = self.encode()
        
        N = self.n_states        
        BL = np.zeros(N-1)  #BL is the block length of compror codewords
    
        h0 = np.log2(np.cumsum([1.0 if sfx == 0 else 0.0 for sfx in self.sfx[1:]]))
        """
        h1 = np.array([h if m == 0 else h+np.log2(m) 
                       for h,m in zip(h0,self.lrs[1:])])
        h1 = np.array([h if m == 0 else h+np.log2(m) 
                       for h,m in zip(h0,self.max_lrs[1:])])
        h1 = np.array([h if m == 0 else h+np.log2(m) 
                       for h,m in zip(h0,self.avg_lrs[1:])])
        """
        h1 = np.array([np.log2(i+1) if m == 0 else np.log2(i+1)+np.log2(m) 
                       for i,m in enumerate(self.max_lrs[1:])])
        
        j = 0
        for i in range(len(code)):
            if self.code[i][0] == 0:
                BL[j] = 1
                j = j+1
            else:
                L = code[i][0]    
                BL[j:j+L] = L #range(1,L+1)
                j = j+L
    
        h1 = h1/BL
        ir = h0 - h1
        ir[ir<0] = 0 #Really a HACK here!!!!!
        return ir, self.code, self.compror
    
    def _ir_cum3(self):
        
        h0 = np.log2(np.cumsum([1.0 if sfx == 0 else 0.0 for sfx in self.sfx[1:]]))
        h1 = np.array([h if m == 0 else (h+np.log2(m))/m 
                       for h,m in zip(h0,self.lrs[1:])])
        
        ir = h0 - h1
        ir[ir<0] = 0 #Really a HACK here!!!!!
        return ir, self.code, self.compror        
    
    def IR(self, alpha = 1.0, ir_type = 'cum'):
        if ir_type == 'cum':
            ir, _code, _compror = self._ir_cum(alpha)
            return ir
        elif ir_type == 'all':
            ir, _code, _compror = self._ir(alpha)
            return ir[1]
        elif ir_type == 'fixed':
            ir, _code, _compror = self._ir_fixed()
            return ir
        elif ir_type == 'cum2':
            ir, _code, _compror = self._ir_cum2()
            return ir
        elif ir_type == 'cum3':
            ir, _code, _compror = self._ir_cum3()
            return ir
    
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
        
        if self.lrs[i] > self.max_lrs[i-1]:
            self.max_lrs.append(self.lrs[i])
        else:
            self.max_lrs.append(self.max_lrs[i-1])
        
        self.avg_lrs.append(self.avg_lrs[i-1]*((i-1.0)/(self.n_states-1.0)) + 
                            self.lrs[i]*(1.0/(self.n_states-1.0)))
    
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
        self.f_array = [0]
        self.data[0] = None
        self.latent = []
        
        # including connectivity
        self.con = []
    
    def reset(self, **kwargs):
        super(MO, self).reset(**kwargs)

        self.kind = 'a'
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
                self.trn[k].append(i) # Create a new forward link to unvisited state
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
        
#         if self.lrs[i] > self.max_lrs:
#             self.max_lrs = self.lrs[i]

        if self.lrs[i] > self.max_lrs[i-1]:
            self.max_lrs.append(self.lrs[i])
        else:
            self.max_lrs.append(self.max_lrs[i-1])
        
        self.avg_lrs.append(self.avg_lrs[i-1]*((i-1.0)/(self.n_states-1.0)) + 
                            self.lrs[i]*(1.0/(self.n_states-1.0)))

            
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
        
        while k != None:
            # NEW Implementation
            if self.params['dfunc'] == 'euclidean':
                a = np.array(new_data) - np.array([self.centroid[self.data[t]] for t in self.trn[k]])
                dvec = np.sqrt((a*a).sum(axis=1))
            elif self.params['dfunc'] == 'other':
                dvec = self.dfunc_handle(new_data, [self.centroid[self.data[t]] for t in self.trn[k]])
                
            # if no transition from suffix
            I = find(dvec < self.params['threshold'])
            if len(I) == 0:
                self.trn[k].append(i) # Create a new forward link to unvisited state
                trn_list.append(k)
                pi_1 = k
                k = self.sfx[k]
            else:
                suffix_candidate = self.trn[k][I[np.argmin(dvec[I])]]
                trn_list.append(i-1)
                break

        if k == None:
            self.sfx[i] = 0
            self.lrs[i] = 0
            self.latent.append([i])
            self.hist.append(1)
            self.data.append(len(self.latent)-1)
            self.centroid.append(new_data)
            if i > 1:
                self.con[self.data[i-1]].add(self.data[i]) 
                self.con.append(set([self.data[i]]))
            else:
                self.con.append(set([]))
        else:
            self.sfx[i] = suffix_candidate
            self.lrs[i] = self._len_common_suffix(pi_1, self.sfx[i]-1) + 1
            _i = self.data[self.sfx[i]]
            self.latent[_i].append(i)
            self.hist[_i] += 1
            self.data.append(_i)
            self.centroid[_i] = (self.centroid[_i] * (self.hist[_i]-1) + new_data)/self.hist[_i]
            self.con[self.data[i-1]].add(self.data[i])
            map(set.add, [self.con[self.data[c]] for c in trn_list], [self.data[i]]*len(trn_list))
        self.rsfx[self.sfx[i]].append(i)
        
#         if self.lrs[i] > self.max_lrs:
#             self.max_lrs = self.lrs[i]        
        if self.lrs[i] > self.max_lrs[i-1]:
            self.max_lrs.append(self.lrs[i])
        else:
            self.max_lrs.append(self.max_lrs[i-1])

        self.avg_lrs.append(self.avg_lrs[i-1]*((i-1.0)/(self.n_states-1.0)) + 
                            self.lrs[i]*(1.0/(self.n_states-1.0)))
                 
                       
def _entropy(x):
    x = np.divide(x, sum(x), dtype = float)
    return sum(np.multiply(-np.log2(x),x))

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

def _build_factor_oracle(oracle, input_data):
    if type(input_data) != np.ndarray or type(input_data[0]) != np.ndarray:
        input_data = np.array(input_data)
            
    for obs in input_data:
        oracle.add_state(obs)
    return oracle
 
def build_oracle(input_data, flag, threshold = 0, feature = None, weights = None, dfunc = 'euclidean', dfunc_handle = None):
    
    # initialize weights if needed 
    if weights == None:
        weights = {}
        weights.setdefault(feature, 1.0)

    if flag == 'a' or flag == 'f' or flag == 'v':
        oracle = _create_oracle(flag, threshold = threshold, weights = weights, dfunc = dfunc, dfunc_handle = dfunc_handle)
    else:
        oracle = _create_oracle('a', threshold = threshold, weights = weights, dfunc = dfunc, dfunc_handle = dfunc_handle)
             
    oracle = _build_factor_oracle(oracle, input_data)
    return oracle 
    

def find_threshold(input_data, r = (0,1,0.1), flag = 'v', feature = None, ir_type='cum', dfunc ='euclidean', dfunc_handle = None, VERBOSE = False):
    thresholds = np.arange(r[0], r[1], r[2])
    irs = []
    for t in thresholds:
        if VERBOSE:
            print 'testing threshold:', t
        tmp_oracle = build_oracle(input_data, flag = flag, threshold = t, feature = feature, dfunc = dfunc, dfunc_handle = dfunc_handle)
        tmp_ir = tmp_oracle.IR(ir_type = ir_type)
        # is it a sum?
        irs.append(tmp_ir.sum())
    # now pair irs and thresholds in a vector, and sort by ir
    ir_thresh_pairs = [(a,b) for a, b in zip(irs, thresholds)]
    pairs_return = ir_thresh_pairs
    ir_thresh_pairs = sorted(ir_thresh_pairs, key= lambda x: x[0], reverse = True)
    return ir_thresh_pairs[0], pairs_return     
    
    