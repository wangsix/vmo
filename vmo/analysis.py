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

import numpy as np
import scipy.spatial.distance as dist
import sys
import itertools

def create_selfsim(oracle, method = 'compror'): 
    """ Create self similarity matrix from compror codes or suffix links
    """
    len_oracle = oracle.n_states - 1
    mat = np.zeros((len_oracle, len_oracle))
    if method == 'compror':
        if oracle.code == []:
            print "Codes not generated. Generating codes with encode()."
            oracle.encode()
        ind = 0
        inc = 1
        for l, p in oracle.code:
            if l == 0:
                inc = 1
            else:
                inc = l
            if inc >= 1:
                for i in range(l):
                    mat[ind+i][p+i-1] = 1
                    mat[p+i-1][ind+i] = 1
            ind = ind + inc
    elif method == 'suffix':
        for i, s in enumerate(oracle.sfx):
            while s != None and s != 0:
                mat[i-1][s-1] = 1
                mat[s-1][i-1] = 1 
                s = oracle.sfx[s] 
    elif method == 'rsfx':
        latent = infer_latent_var(oracle)
        for i in range(len(latent)):
            _l = latent[i]
            p = itertools.product(_l, repeat = 2)
            for _p in p:
                mat[_p[0]-1][_p[1]-1] = 1            
    return mat

def create_transition(oracle, method = 'trn'):
    if oracle.kind == 'r':
        mat, hist = _create_trn_mat_symbolic(oracle, method)
        return mat, hist
    elif oracle.kind ==  'a':
        raise NotImplementedError("Audio version is under construction, coming soon!")

def _create_trn_mat_symbolic(oracle, method):
    n = oracle.get_num_symbols()
    sym_list = [oracle.data[_s] for _s in oracle.trn[0]]
    hist = np.zeros((n))
    mat = np.zeros((n,n))
    for i in range(1, oracle.n_states-1):
        _i = sym_list.index(oracle.data[i])
        if method == 'trn':
            trn_list = oracle.trn[i]
#         elif method == 'rsfx':
#             trn_list = oracle.rsfx[i]
        elif method == 'seq':
            trn_list = [i+1]

        for j in trn_list:
            if j < oracle.n_states:
                _j = sym_list.index(oracle.data[j])
                mat[_i][_j] += 1
            else: 
                print "index " + str(j) + " is out of bounds." 
            hist[_i] += 1
    mat = mat.transpose()/hist
    mat = mat.transpose()
    return mat, hist

def _test_context(oracle, context):
    _b, _s = oracle.accept(context)
    while not _b:
        context = context[1:]
        _b, _s = oracle.accept(context)
    return _b, _s, context

def predict(oracle, context, ab=[], VERBOSE = 0):
    if VERBOSE:
        print "original context: ", context
    if ab == []:
        ab = oracle.get_alphabet()
        
    _b, _s, context = _test_context(oracle, context)    
    _lrs = [oracle.lrs[k] for k in oracle.rsfx[_s]]
    context_state = []
    while context_state == []:
        for _i,_l in enumerate(_lrs):
            if _l >= len(context):
                context_state.append(oracle.rsfx[_s][_i])
        if context_state != []:
            break
        else:
            context = context[1:]
            _b, _s = oracle.accept(context)
            _lrs =  [oracle.lrs[k] for k in oracle.rsfx[_s]]
    if VERBOSE:                    
        print "final context: ", context
        print "context_state: ", context_state
    d_count = len(ab) 
    hist = [1.0] * len(ab) # initialize all histograms with 1s.
    
    trn_data = [oracle.data[n] for n in oracle.trn[_s]]
    for k in trn_data:
        hist[ab[k]] += 1.0
        d_count += 1.0
    
    for i in context_state:
        d_count, hist = _rsfx_count(oracle, i, d_count, hist, ab)
    
    return [hist[idx]/d_count for idx in range(len(hist))], context   
     
def _rsfx_count(oracle, s, count, hist, ab, VERBOSE = 0):
    """ Accumulate counts for context 
    """
    trn_data = [oracle.data[n] for n in oracle.trn[s]]
    for k in trn_data:
        hist[ab[k]] += 1.0
        count += 1.0

    rsfx_candidate = oracle.rsfx[s][:]
    while rsfx_candidate != []:
        s = rsfx_candidate.pop(0)
        trn_data = [oracle.data[n] for n in oracle.trn[s]]
        for k in trn_data:
            hist[ab[k]] += 1.0
            count += 1.0
        rsfx_candidate.extend(oracle.rsfx[s])
        
    return count, hist
                     
def infer_latent_var(oracle):
    """ Return lists of states connected by suffix links"""
    l = []
    for k in oracle.rsfx[0]:
        tmp = []
        tmp.append(k)
        _c = oracle.rsfx[k][:]
        while _c != []:
            _k = _c.pop(0)
            tmp.append(_k)
            _c.extend(oracle.rsfx[_k])
        l.append(tmp)
    return l
            
def logEval(oracle, testSequence, ab = [], m_order = None, VERBOSE = 0):
    ''' Evaluate the average log-loss of a sequence given an oracle 
    '''
    if ab == []:
        ab = oracle.get_alphabet()
    if VERBOSE:
        print ' '
    
    logP = 0.0
    context = []
    increment = np.floor((len(testSequence)-1)/100)
    bar_count = -1
    maxContextLength = 0
    avgContext = 0
    for i,t in enumerate(testSequence):
        
        p, c = predict(oracle, context, ab, VERBOSE = 0)
        if len(c) < len(context):
            context = context[-len(c):]
        logP -= np.log2(p[ab[t]])
        context.append(t)
        
        if m_order != None:
            if len(context) > m_order:
                context = context[-m_order:]
        avgContext += float(len(context))/len(testSequence)

        if VERBOSE:
            percentage = np.mod(i, increment)
            if percentage == 0:
                bar_count += 1
            if len(context) > maxContextLength:
                maxContextLength = len(context)
            sys.stdout.write('\r')
            sys.stdout.write("\r[" + "=" * bar_count +  
                             " " * (100-bar_count) + "] " +  
                             str(bar_count) + "% " +
                             str(i)+"/"+str(len(testSequence)-1)+" Current max length: " + str(maxContextLength))
            sys.stdout.flush()
    return logP/len(testSequence), avgContext
            
            
def cluster(oracle):
    raise NotImplementedError("cluster() is under construction, coming soon!")
    
def segment(oracle):
    raise NotImplementedError("segment() is under construction, coming soon!")

# def _rsfxmin(oracle, n, x, theta, c):
#     if oracle.rsfx[n] == []:
#         return c, theta
#     else:
#         for k in oracle.rsfx[n]:
#             theta_hat = get_distance(x, oracle.feature[k], oracle.params['weights'], oracle.params['dfunc'])
#             if theta > theta_hat:
#                 theta = theta_hat
#                 c = k
#             _c, _theta = _rsfxmin(oracle, k, x, theta, c)
#             if _theta < theta:
#                 theta = _theta
#                 c = _c            
#         return c, theta 
#     
# def _sfxmin(oracle, n, x, theta, c):
#     while oracle.sfx[n] != 0:
#         theta_hat = get_distance(x, oracle.feature[oracle.sfx[n]], oracle.params['weights'], oracle.params['dfunc'])
#         if theta > theta_hat:
#             theta = theta_hat
#             c = oracle.sfx[n]        
#         n = oracle.sfx[n]
#     return c, theta

def _get_sfx(oracle, s_set, k):
    while oracle.sfx[k] != 0:
        s_set.add(oracle.sfx[k])
        k = oracle.sfx[k]
    return s_set

def _get_rsfx(oracle, rs_set, k):
    if oracle.rsfx[k] == []:
        return rs_set
    else:
        rs_set = rs_set.union(oracle.rsfx[k])
        for _k in oracle.rsfx[k]:
            rs_set = rs_set.union(_get_rsfx(oracle, rs_set, _k))
        return rs_set
            
        
def query_complete(oracle, query, method = 'trn', selftrn = True, smooth = False, weight = 0.5):
    """ Return the closest path in target oracle given a query sequence
    
    Args:
        oracle: an oracle object already learned, the target. 
        query: the query sequence in a matrix form such that 
             the ith row is the feature at the ith time point
        method: 
        selftrn:
        smooth:(off-line only)
        weight:
    
    """
    N = len(query)
    K = oracle.get_num_symbols()
    C = [0] * K # cost vector
    P = [[0]* N for _i in range(K)] # path matrix 
    
    if smooth:
        D = dist.pdist(oracle.f_array[1:], 'sqeuclidean')
        D = dist.squareform(D, checks = False)
    
    for ind,k in enumerate(oracle.rsfx[0]): # emission transition
        a = np.array(query[0]) - np.array([oracle.f_array[t] for t in oracle.latent[oracle.data[k]]])
        dvec = (a*a).sum(axis=1) # Could skip the sqrt
        P[ind][0] = oracle.latent[oracle.data[k]][dvec.argmin()]
        C[ind] += dvec.min()
        
    for i in range(1,N): # iterate over the rest of query
        state_cache = []
        distance_cache = np.array([0.0] * oracle.n_states)
        for k in range(K): # iterate over the K possible paths                
            if method == 'complete':
                _trn = list(itertools.chain.from_iterable([oracle.latent[_c] for _c in list(oracle.con[oracle.data[P[k][i-1]]])]))
            else:
                _trn = oracle.trn[P[k][i-1]][:] # Sub-optimal
                if _trn == []:
                    _trn = oracle.trn[oracle.sfx[P[k][i-1]]][:]
                if selftrn:  
                    _trn.append(P[k][i-1])
            
            _trn_unseen = [_t for _t in _trn if _t not in state_cache]
            state_cache.extend(_trn_unseen)
                                
            t = list(itertools.chain.from_iterable([oracle.latent[oracle.data[j]] for j in _trn]))
#             a = np.array(query[i]) - np.array([oracle.f_array[_t] for _t in t])
            t_unseen = list(itertools.chain.from_iterable([oracle.latent[oracle.data[j]] for j in _trn_unseen]))
            if t_unseen != []:
                a = np.array(query[i]) - np.array([oracle.f_array[_t] for _t in t_unseen])
                dvec = (a*a).sum(axis=1) # Could skip the sqrt
                distance_cache[t_unseen] = dvec
            dvec = distance_cache[t]
            if smooth and P[k][i-1] < oracle.n_states-1:
                dvec = dvec * (1.0-weight) + weight*np.array([D[_t-1][P[k][i-1]] for _t in t])
            P[k][i] = t[dvec.argmin()]
            C[k] += dvec.min()
                          
    i_hat = np.argmin(C)
    return P, C, i_hat    


