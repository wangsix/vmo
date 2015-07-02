"""
utils/prism/model.py
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
import string

import vmo.utils.probabilities as probas

"""Graph to PRISM input conversion function"""

def print_proba(proba):
    """Print a probability, with output depending on the input's style."""
    if isinstance(proba, Fraction):
        return "{0}/{1}".format(proba.numerator, proba.denominator)
    elif isinstance(proba, float) or isinstance(proba, int):
        print_proba(Fraction(proba))
    else:
        raise ValueError("Unexpected probability type")
                
def print_transitions(proba_adj_lists, origin, prism_state_name='s',
                      make_update_string=(lambda **kwargs : "")):
    """Return a list of PRISM-format bytearrays for all transitions from origin.

    Keyword arguments:
        proba_adj_lists: list of lists of pairs
            A probabilistic graph, given as it adjacency list
        origin_state: int
            The starting state in the graph
        prism_state_name: string
            The name of the state in the PRISM model which is
            represented by the graph (default 's')
        make_update_string: function
            A generic function to print additional updates (such as a
            constant value assignation) within the transition
            (default <print an empty string>)
    ----
    Two-states a & b, a leads to b and b leads to a
    >>> graph = [[1], [0]]
    >>> proba_graph = probas.uniform(graph)
    >>> result = print_transitions(proba_graph, 0)
    >>> expected = [bytearray("1/1 : (s'=1)")]
    >>> result == expected
    True
    """
    goals = proba_adj_lists[origin]
    transitions = []
        
    for (goal, proba) in goals:
        new_transition = "{0} : ({1}'={2})".format(
            print_proba(proba), prism_state_name, goal)
        update_string = make_update_string(goal=goal, proba=proba)
        bytearray(new_transition).extend(update_string)
        transitions.append(new_transition)

    return transitions

def print_state(proba_adj_lists, s_index, prism_state_name='s'):
    """Return a PRISM-formatted bytearray defining state s_index.

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
    transitions_str = bytearray(" + ").join(transitions) 
    return _bytes_concat([guard, transitions_str])

def print_graph(proba_adj_lists, init_state=0, prism_state_name='s'):
    """Return a list of PRISM-formatted bytearray defining all graph's states.

    Keyword arguments:
        proba_adj_lists: list of lists of pairs
            A probabilistic graph, given as it adjacency list
        init_state: int
            The index of the DTMC's initial state (default 0)
        prism_state_name: string
            The name of the state in the PRISM model which is
            represented by the graph (default 's') 
    ----
    Three-states (a, b and c), a leads to b or c, b and c lead to a
    >>> graph = [[1, 2], [0], [0]]
    >>> proba_graph = probas.uniform(graph)
    >>> result = print_graph(proba_graph, prism_state_name='a')
    >>> expected = (map(bytearray, [
    ...      "// local state",
    ...      "a : [0..2] init 0"]
    ...      ),
    ...      map(bytearray, [
    ...      "[] a=0 -> 1/2 : (a\'=1) + 1/2 : (a\'=2)",
    ...      "[] a=1 -> 1/1 : (a\'=0)",
    ...      "[] a=2 -> 1/1 : (a\'=0)"]
    ...     ))
    >>> result == expected
    True
    """
    states_num = len(proba_adj_lists)
    header = []

    # Esthetic comment-line
    header.append(bytearray("// local state"))
    # Declaration of the initial state and the number of states
    range_init = "{0} : [0..{1}] init {2}".format(prism_state_name,
                                                  states_num-1,
                                                  init_state)
    header.append(bytearray(range_init))
    
    states = []
    for state in range(states_num):
        new_state_bytes = print_state(proba_adj_lists, state,
                                      prism_state_name=prism_state_name)
        states.append(new_state_bytes)
    
    return header, states

def print_dtmc_module(proba_adj_lists, prism_state_name='s', module_name='m'):
    """Return a PRISM-formatted bytearray defining a dtmc-module for the graph.

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
    ...     "\\t// local state;\\n" +
    ...     "\\ta : [0..2] init 0;\\n" +
    ...     "\\t\\n" +
    ...     "\\t[] a=0 -> 1/2 : (a\'=1) + 1/2 : (a\'=2);\\n" +
    ...     "\\t[] a=1 -> 1/1 : (a\'=0);\\n" +
    ...     "\\t[] a=2 -> 1/1 : (a\'=0);\\n" +
    ...     "\\n" +
    ...     "endmodule\\n"
    ...     )
    >>> result == expected
    True
    """
    header = bytearray("dtmc\n\n" + "module {0}\n\n".format(module_name))
    footer = bytearray("\nendmodule\n")
    
    graph_header, graph_states = print_graph(
        proba_adj_lists, prism_state_name=prism_state_name)
    graph_header_str = indent_join_lines(graph_header)
    graph_states_str = indent_join_lines(graph_states)

    graph_str = _bytes_concat([graph_header_str,
                               bytearray("\t\n"),
                               graph_states_str])
    
    return _bytes_concat([header, graph_str, footer])

"""Auxiliary functions"""

def indent_join_lines(lines, tabulate_by=1, trailing=';\n'):
    """Return the concatenation of lines with inserted tabs and newlines.
    
    Keyword arguments:
        lines: list of strings
            The lines to concatenate
        tabulate_by: int
            The number of tabulations to add at te beginning of every line
        trailing: str
            A string to append to all lines 
    """
    tabulation = "\t" * tabulate_by
    tabulate_line = lambda line : tabulation + line + trailing
    tabbed_lines = map(tabulate_line, lines)
    return _bytes_concat(tabbed_lines)

def _bytes_concat(bytearrays):
    """Efficiently concatenate the input bytearrays.

    ----
    >>> bytes = bytearray("Hello !")
    >>> _bytes_concat([bytes]*2) == bytearray("Hello !Hello !")
    True
    """ 
    return bytearray("").join(bytearrays)    
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()
