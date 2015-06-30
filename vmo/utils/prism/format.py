"""
utils/prism/format.py
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
"""

import numpy as np
from fractions import Fraction
import vmo.logics.model_checking.probabilities as probas

"""Oracle to PRISM input conversion function"""

def print_proba(proba):
    """Print a probability, with output depending on the input's style."""
    if isinstance(proba, Fraction):
        return "{0}/{1}".format(proba.numerator, proba.denominator)
    elif isinstance(proba, float) or isinstance(proba, int):
        print_proba(Fraction(proba))
    else:
        raise ValueError("Unexpected probability type")
                
def print_transitions(proba_adj_lists, origin, prism_state_name='s'):
    """Print a PRISM-formatted bytearray of all transitions from origin.

    Keyword arguments:
        proba_adj_lists: list of lists of pairs
            A probabilistic graph, given as it adjacency list
        origin_state: int
            The starting state in the graph
        prism_state_name: string
            The name of the state in the PRISM model which is
            represented by the graph (default 's') 
    ----
    Two-states a & b, a leads to b and b leads to a
    >>> graph = [[1], [0]]
    >>> proba_graph = probas.uniform(graph)
    >>> result = print_transitions(proba_graph, 0)
    >>> expected = bytearray("1/1 : (s'=1)")
    >>> result == expected
    True
    """
    goals = proba_adj_lists[origin]
    transitions = bytearray("")
        
    for (goal, proba) in goals[:-1]:
        new_transition = "{0} : ({1}'={2}) + ".format(
            print_proba(proba), prism_state_name, goal)
        transitions.extend(new_transition)

    # Operate on last transition, don't print " + " connector
    last_goal, proba = goals[-1]
    last_transition = "{0} : ({1}'={2})".format(
        print_proba(proba), prism_state_name, last_goal)
    transitions.extend(last_transition)
    
    return transitions

def print_state(proba_adj_lists, s_index, prism_state_name='s'):
    """Print a PRISM-formatted bytearray defining state s_index.

    Keyword arguments:
        proba_adj_lists: list of lists of pairs
            A probabilistic graph, given as it adjacency list
        s_index: int
            The state in the graph to print
        prism_state_name: string
            The name of the state in the PRISM model which is
            represented by the graph (default 's') 
    ----
    Two-states a & b, a leads to b and b leads to a
    >>> graph = [[1], [0]]
    >>> proba_graph = probas.uniform(graph)
    >>> result = print_state(proba_graph, 0)
    >>> expected = bytearray("[] s=0 -> 1/1 : (s'=1)")
    >>> result == expected
    True
    """
    guard = bytearray("[] {0}={1} -> ".format(prism_state_name, s_index))
    transitions = print_transitions(proba_adj_lists, s_index,
                                    prism_state_name)
    return guard + transitions

def print_graph(proba_adj_lists, prism_state_name='s'):
    """Print a PRISM-formatted bytearray defining the input graph.

    Keyword arguments:
        proba_adj_lists: list of lists of pairs
            A probabilistic graph, given as it adjacency list
        prism_state_name: string
            The name of the state in the PRISM model which is
            represented by the graph (default 's') 
    ----
    Three-states (a, b and c), a leads to b or c, b and c lead to a
    >>> graph = [[1, 2], [0], [0]]
    >>> proba_graph = probas.uniform(graph)
    >>> result = print_graph(proba_graph, prism_state_name='a')
    >>> expected = bytearray(
    ...     "// local state\\n" +
    ...     "a : [0..2] init 0;\\n\\n" +
    ...     "[] a=0 -> 1/2 : (a\'=1) + 1/2 : (a\'=2);\\n" +
    ...     "[] a=1 -> 1/1 : (a\'=0);\\n" +
    ...     "[] a=2 -> 1/1 : (a\'=0);\\n"
    ...     )
    >>> result == expected
    True
    """
    states_num = len(proba_adj_lists)
    header = bytearray(
        "// local state\n" +
        "{0} : [0..{1}] init 0;\n\n".format(prism_state_name, states_num-1)
        )

    states = bytearray("")
    for state in range(states_num):
        new_state_bytes = print_state(proba_adj_lists, state,
                                      prism_state_name=prism_state_name)
        new_state_bytes.extend(";\n")
        states.extend(new_state_bytes)

    return header + states

def print_dtmc_module(proba_adj_lists, prism_state_name='s', module_name='m'):
    """Print a PRISM-formatted bytearray defining a dtmc-module for the graph.

    Keyword arguments:
        proba_adj_lists: list of list of pairs
            A probabilistic graph, given as it adjacency list
        prism_state_name: string
            The name of the state in the PRISM model which is
            represented by the graph (default 's') 
    ----
    Three-states (a, b and c), a leads to b or c, b and c lead to a
    >>> graph = [[1, 2], [0], [0]]
    >>> proba_graph = probas.uniform(graph)
    >>> result = print_dtmc_module(proba_graph, prism_state_name='a',
    ...                            module_name='test')
    >>> expected = bytearray(
    ...     "dtmc\\n" +
    ...     "\\n" +
    ...     "module test\\n" +
    ...     "\\n" +
    ...     "\\t// local state\\n" +
    ...     "\\ta : [0..2] init 0;\\n" +
    ...     "\\t\\n" +
    ...     "\\t[] a=0 -> 1/2 : (a\'=1) + 1/2 : (a\'=2);\\n" +
    ...     "\\t[] a=1 -> 1/1 : (a\'=0);\\n" +
    ...     "\\t[] a=2 -> 1/1 : (a\'=0);\\n" +
    ...     "\\n" +
    ...     "endmodule"
    ...     )
    >>> result == expected
    True
    """
    header = bytearray("dtmc\n\n" + "module {0}\n\n".format(module_name))
    footer = bytearray("\nendmodule")

    graph_str = print_graph(proba_adj_lists, prism_state_name=prism_state_name)

    newlines_count = graph_str.count('\n')
    graph_str = graph_str.replace('\n', '\n\t', newlines_count-1)
    graph_str = '\t' + graph_str
    
    return header + graph_str + footer
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()
