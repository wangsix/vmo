"""
utils/probabilities.py
Variable Markov Oracle in python

@copyright: 
Copyright (C) 7.2015 Theis Bazin

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
"""

import numpy as np
from fractions import Fraction

"""Various probability laws to transform a graph into a probabilitic graph"""

def uniform(adj_lists):
    """Return a probabilistic graph with uniform transition probabilities.

    Keyword arguments:
        adj_lists: list of lists
            The original, non-probabilistic graph
    ----
    Two-states a & b, a leads to b and b leads to a
    >>> graph = [[1], [0]]
    >>> proba_graph = uniform(graph)
    >>> one_frac = Fraction(1, 1)
    >>> proba_graph == [[(1, one_frac)], [(0, one_frac)]]
    True
    """ 
    proba_transitions = []
    for neighbours in adj_lists:
        count = len(neighbours)
        if count == 0:
            proba_transitions.append([])
        else:
            # Equidistributed-probability amongst all neighbours
            uniform_proba = Fraction(numerator=1, denominator=count)
            
            # Label destination state with the transition's probability
            to_proba = lambda state : (state, uniform_proba)
            
            proba_neighbours = map(to_proba, neighbours)
            proba_transitions.append(proba_neighbours)
    return proba_transitions

if __name__ == "__main__":
    import doctest
    doctest.testmod()
