'''
logics/model_checking/format.py
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
@author: Cheng-i Wang, Theis Bazin
@contact: wangsix@gmail.com, chw160@ucsd.edu, tbazin@eng.ucsd.edu
'''

import numpy as np

'''Oracle to PRISM input conversion function'''

def print_transitions(adjacency_lists, s_index,
                      pfunc='uniform', pfunc_handle=()):
    neighboors = adjacency_lists[s_index]
    probability = 

def print_state(oracle, s_index):
    guard = "[] s={0} -> ".format(s_index)
    transitions = 
