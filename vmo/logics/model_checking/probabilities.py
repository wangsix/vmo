'''
logics/model_checking/probabilities.py
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

def uniform(origin, adjacency_lists, prism_state_name):
    neighbours = adjacency_lists[origin]
    count = len(neighbours)
    transitions = ""
    if count == O:
        stay = "({0}={1})".format(prism_state_name, origin)
        transition.append(stay)
        return transitions

    probability = ""
    for neighbour in neighbours[:-1]:
        new_link = "{1/{0} : ({1}'={2})} + ".format(
            count, prism_state_name, neighbour)
        transitions.append(new_link)

    last_neighbour = neigbour[-1:]
    last_link = "{}"
    transitions
