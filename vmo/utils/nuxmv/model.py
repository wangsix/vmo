"""
utils/nuxmv/model.py
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

import vmo.analysis as van
import vmo.utils.chromagram as chroma
import vmo.utils.probabilities as probas

"""Graph to nuXmv input conversion function"""
                
def print_transitions(adj_lists, origin):
    """Return a compact nuXmv set countaining all of <origin>'s neighbours.

    Keyword arguments:
        adj_lists: list of lists of ints
            A graph, given as it adjacency list
        origin_state: int
            The starting state in the graph
    ----
    Two-states a & b, a leads to b and b leads to a
    >>> adj_lists = [[1, 2], [0], [1]]
    >>> result = print_transitions(adj_lists, 0)
    >>> expected = bytearray("{1, 2}")
    >>> result == expected
    True
    """
    goals = sorted(adj_lists[origin])

    to_bytearray = lambda state: bytearray(str(state))
    goals_str = bytearray(", ").join(map(to_bytearray, goals))

    nuxmv_set = "{{{0}}}".format(goals_str)
    
    return bytearray(nuxmv_set)

def print_state(adj_lists, s_index, nuxmv_state_name='s',
                extended_guard=None):
    """Return a nuXmv-formatted bytearray defining state s_index.
    
    Keyword arguments:
        adj_lists: list of lists of pairs
            A probabilistic graph, given as it adjacency list
        s_index: int
            The state in the graph to print
        nuxmv_state_name: string
            The name of the state in the nuXmv model which is
            represented by the graph (default 's')
        extended_guard: callable
            A function to generate more complex guards
            (default None)
    ----
    Two-states a & b, a leads to b and b leads to a
    >>> adj_lists = [[1], [0]]
    >>> result = print_state(adj_lists, 0)
    >>> expected = bytearray("s=0: {1};")
    >>> result == expected
    True
    """
    if extended_guard is None:
        extended_guard_str = ''
    else:
        extended_guard_str = extended_guard(adj_lists, s_index,
                                            nuxmv_state_name) 
    
    guard = bytearray("{0}={1}{2}: ".format(nuxmv_state_name, s_index,
                                            extended_guard_str))
    transitions_str = print_transitions(adj_lists, s_index)
    return _bytes_concat([guard, transitions_str, bytearray(';')])

def print_graph(adj_lists, init_state=0, nuxmv_state_name='s'):
    """Return a list of nuXmv-formatted bytearrays defining all graph's states.

    Keyword arguments:
        adj_lists: list of lists of pairs
            A probabilistic graph, given as it adjacency list
        init_state: int
            The index of the DTMC's initial state (default 0)
        nuxmv_state_name: string
            The name of the state in the nuXmv model which is
            represented by the graph (default 's') 
    ----
    Three-states (a, b and c), a leads to b or c, b and c lead to a
    >>> adj_lists = [[1, 2], [0], [0]]
    >>> result = print_graph(adj_lists, nuxmv_state_name='a')
    >>> expected = (map(bytearray, [
    ...      "VAR a: 0..2;",
    ...      "ASSIGN init(a) := 0;"]
    ...      ),
    ...      map(bytearray, [
    ...      "next(a) :=",
    ...      "case",
    ...      "a=0: {1, 2};",
    ...      "a=1: {0};",
    ...      "a=2: {0};",
    ...      "esac;"])
    ...      )
    >>> result == expected
    True
    """
    states_num = len(adj_lists)
    header = []

    # Declare number of states
    range_decl = "VAR {0}: 0..{1};".format(nuxmv_state_name,
                                           states_num-1)
    header.append(bytearray(range_decl))
    # Declare initial state
    init_state = "ASSIGN init({0}) := {1};".format(nuxmv_state_name,
                                                 init_state)
    header.append(bytearray(init_state))
    
    states = []
    states.append(bytearray("next({}) :=".format(nuxmv_state_name)))
    states.append(bytearray("case"))

    for state in range(states_num):
        new_state_bytes = print_state(adj_lists, state,
                                      nuxmv_state_name=nuxmv_state_name)
        states.append(new_state_bytes)
    states.append(bytearray("esac;"))
    
    return header, states

def print_module(adj_lists, nuxmv_state_name='s', module_name='main'):
    """Return a nuXmv-formatted bytearray defining a dtmc-module for the graph.

    Keyword arguments:
        adj_lists: list of list of pairs
            A probabilistic graph, given as it adjacency list
        nuxmv_state_name: string
            The name of the state in the nuXmv model which is
            represented by the graph (default 's') 
    ----
    Three-states (a, b and c), a leads to b or c, b and c lead to a
    >>> adj_lists = [[1, 2], [0], [0]]
    >>> result = print_module(adj_lists, nuxmv_state_name='a',
    ...                       module_name='test')
    >>> expected = bytearray(
    ...     "MODULE test()\\n" +
    ...     "VAR a: 0..2;\\n" +
    ...     "ASSIGN init(a) := 0;\\n" +
    ...     "\\tnext(a) :=\\n" +
    ...     "\\tcase\\n" +
    ...     "\\ta=0: {1, 2};\\n" +
    ...     "\\ta=1: {0};\\n" +
    ...     "\\ta=2: {0};\\n" +
    ...     "\\tesac;\\n"
    ...     )
    >>> result == expected
    True
    """
    header = bytearray("MODULE {0}()\n".format(module_name))
    
    graph_header, graph_states = print_graph(
        adj_lists, nuxmv_state_name=nuxmv_state_name)
    graph_header_str = indent_join_lines(graph_header, tabulate_by=0)
    graph_states_str = indent_join_lines(graph_states) 

    graph_str = _bytes_concat([graph_header_str,
                               graph_states_str])
    
    return _bytes_concat([header, graph_str])

"""Harmonic printing"""

"""Prints a nuXmv mapping from oracle states to pitch-space."""

def print_pitch_state(oracle, state, nuxmv_state_name='s'):
    """Return a nuXmv assignation case to the root of `state`'s chord.

    Assumes the oracle has been created with a chromagram as feature.
    ----
    >>> import music21
    >>> import vmo
    >>> c1 = music21.chord.Chord(['C4', 'E4', 'G4'], quarterLength=1)
    >>> c2 = music21.chord.Chord(['E4', 'G4', 'B4'], quarterLength=1)
    >>> s = music21.stream.Stream([c1])
    >>> s.append(c2)
    >>> chromagram = chroma.from_stream(s)
    >>> o = vmo.VMO.oracle.build_oracle(chromagram.T, 'a',
    ...                                 feature='chromagram')
    >>> result = print_pitch_state(o, 1)
    >>> expected = bytearray("s=1: p_C;")
    >>> result == expected
    True
    """
    chroma_array = oracle.feature[state]
    chord = chroma.to_chord(chroma_array)
    root_name = ('p_' + chord.root().name)
    return bytearray("{0}={1}: {2};".format(nuxmv_state_name, state,
                                           root_name))

def print_pitches(oracle, nuxmv_state_name='s'):
    header = []
    pitch_decl = ("VAR pitchRoot : {None, p_C, p_C#, p_D, p_E-, p_E, p_F, " +
                  "p_F#, p_G, p_G#, p_A, p_B-, p_B};")
    header.append(bytearray(pitch_decl))
    pitch_init = "ASSIGN init(pitchRoot) := None;"
    header.append(bytearray(pitch_init))

    cases = []
    cases.append(bytearray("next(pitchRoot) :="))
    cases.append(bytearray("case"))
    cases.append(bytearray("{}=0: None;".format(nuxmv_state_name)))
    for s in range(oracle.n_states)[1:]:
        cases.append(print_pitch_state(oracle, s, nuxmv_state_name))
    cases.append(bytearray("esac;"))

    return header, cases

"""Print chromagram oracle"""

def print_oracle(oracle, nuxmv_state_name='s'):
    """Return a bytearray describing `oracle`, with oracle states and pitches.

    Assumes the oracle has been created with a chromagram as feature.
    ----
    >>> import music21
    >>> import vmo
    >>> c1 = music21.chord.Chord(['C4', 'E4', 'G4'], quarterLength=1)
    >>> c2 = music21.chord.Chord(['E4', 'G4', 'B4'], quarterLength=1)
    >>> s = music21.stream.Stream([c1])
    >>> s.append(c2)
    >>> chromagram = chroma.from_stream(s)
    >>> o = vmo.VMO.oracle.build_oracle(chromagram.T, 'a',
    ...                                 feature='chromagram')
    >>> result = print_oracle(o)
    >>> expected = bytearray(
    ...     "MODULE main()\\n" +
    ...     "VAR s: 0..2;\\n" +
    ...     "ASSIGN init(s) := 0;\\n" +
    ...     "\\tnext(s) :=\\n" +
    ...     "\\tcase\\n" +
    ...     "\\ts=0: {1, 1, 2, 2};\\n" +
    ...     "\\ts=1: {0, 2};\\n" +
    ...     "\\ts=2: {0};\\n" +
    ...     "\\tesac;\\n\\n" +
    ...     "VAR pitchRoot : {None, p_C, p_C#, p_D, p_E-, p_E, " +
    ...     "p_F, p_F#, p_G, p_G#, p_A, p_B-, p_B};\\n" +
    ...     "ASSIGN init(pitchRoot) := None;\\n" +
    ...     "\\tnext(pitchRoot) :=\\n" +
    ...     "\\tcase\\n" +
    ...     "\\ts=0: None;\\n" +
    ...     "\\ts=1: p_C;\\n" +
    ...     "\\ts=2: p_E;\\n" +
    ...     "\\tesac;\\n"
    ...     )
    >>> result == expected
    True
    """
    adj_lists = van.graph_adjacency_lists(oracle)
    base_model = print_module(adj_lists, nuxmv_state_name)

    pitches_header, pitches_cases = print_pitches(oracle, nuxmv_state_name)
    pitches_header_str = indent_join_lines(pitches_header, tabulate_by=0)
    pitches_cases_str = indent_join_lines(pitches_cases) 

    pitches_str = _bytes_concat([pitches_header_str,
                               pitches_cases_str])
    
    return _bytes_concat([base_model, bytearray("\n"), pitches_str])

        
"""Auxiliary functions"""

def indent_join_lines(lines, tabulate_by=1, trailing='\n'):
    """Return the concatenation of `lines` with inserted tabs and newlines.
    
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

    Keyword arguments:
        bytearrays: list of byetarray
            The bytearrays to concatenate
    ----
    >>> bytes = bytearray("Hello !")
    >>> _bytes_concat([bytes]*2) == bytearray("Hello !Hello !")
    True
    """ 
    return bytearray("").join(bytearrays)    
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()