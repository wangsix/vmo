"""
utils/nuxmv/model.py
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
import string

import music21 as mus

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
    Two-states a and b, a leads to b and b leads to a.
    >>> adj_lists = [[1, 2], [0], [1]]
    >>> result = print_transitions(adj_lists, 0)
    >>> expected = "{1, 2}"
    >>> result == expected
    True
    """
    goals = sorted(adj_lists[origin])

    goals_str = ", ".join([str(goal) for goal in goals])

    nuxmv_set = "{{{0}}}".format(goals_str)
    
    return nuxmv_set

def print_state(adj_lists, s_index, nuxmv_state_name='s',
                extended_guard=None):
    """Return a nuXmv-formatted string defining state s_index.
    
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
    Two-states a and b, a leads to b and b leads to a.
    >>> adj_lists = [[1], [0]]
    >>> result = print_state(adj_lists, 0)
    >>> expected = "s=0: {1};"
    >>> result == expected
    True
    """
    if extended_guard is None:
        extended_guard_str = ''
    else:
        extended_guard_str = extended_guard(adj_lists, s_index,
                                            nuxmv_state_name) 
    
    guard = "{0}={1}{2}: ".format(nuxmv_state_name, s_index,
                                  extended_guard_str)
    transitions_str = print_transitions(adj_lists, s_index)
    return "".join([guard, transitions_str, ';'])

def print_graph(adj_lists, init_state=0, nuxmv_state_name='s'):
    """Return a list of nuXmv-formatted strings defining all graph's states.

    Keyword arguments:
        adj_lists: list of lists of pairs
            A probabilistic graph, given as it adjacency list
        init_state: int
            The index of the DTMC's initial state (default 0)
        nuxmv_state_name: string
            The name of the state in the nuXmv model which is
            represented by the graph (default 's') 
    ----
    Three-states (a, b and c), a leads to b or c, b and c lead to a.
    >>> adj_lists = [[1, 2], [0], [0]]
    >>> result = print_graph(adj_lists, nuxmv_state_name='a')
    >>> expected = (
    ...      ["VAR a: 0..2;",
    ...       "ASSIGN init(a) := 0;"],
    ...      ["next(a) :=",
    ...       "case",
    ...       "a=0: {1, 2};",
    ...       "a=1: {0};",
    ...       "a=2: {0};",
    ...       "esac;"])
    ...      )
    >>> result == expected
    True
    """
    states_num = len(adj_lists)
    header = []

    # Declare number of states
    range_decl = "VAR {0}: 0..{1};".format(nuxmv_state_name,
                                           states_num-1)
    header.append(range_decl)
    # Declare initial state
    init_state = "ASSIGN init({0}) := {1};".format(nuxmv_state_name,
                                                   init_state)
    header.append(init_state)
    
    states = []
    states.append("next({}) :=".format(nuxmv_state_name))
    states.append("case")

    for state in range(states_num):
        new_state = print_state(adj_lists, state,
                                nuxmv_state_name=nuxmv_state_name)
        states.append(new_state)
    states.append("esac;")
    
    return header, states

def print_module(adj_lists, nuxmv_state_name='s', init_state=0,
                 module_name='main'):
    """Return a nuXmv-formatted string defining a dtmc-module for the graph.

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
    >>> expected = (
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
    header = "MODULE {0}()\n".format(module_name)
    
    graph_header, graph_states = print_graph(adj_lists,
                                             init_state=init_state,
                                             nuxmv_state_name=nuxmv_state_name)
    graph_header_str = indent_join_lines(graph_header, tabulate_by=0)
    graph_states_str = indent_join_lines(graph_states) 

    graph_str = "".join([graph_header_str,
                         graph_states_str])
    
    return "".join([header, graph_str])

def print_variable(name, values, initial_value, transitions):
    """Generic formatted variable declaration.

    Keyword arguments:
        name: str
            The variable's name.
        values: str
            The variable's type / range.
        initial_value: str
            The variable's initial value.
        transitions: list of strings
            The list of all transitions for the variable.
    ----
    >>> name = 'var'
    >>> values = '{a, b}'
    >>> initial_value = 'a'
    >>> transitions = ["var=a: b;",
    ...                "var=b: a;"]
    >>> expected = (["VAR var : {a, b};",
    ...              "ASSIGN init(var) := a;"],
    ...             ["next(var) :=",
    ...              "case",
    ...              "var=a: b;",
    ...              "var=b: a;",
    ...              "esac;"])
    >>> result = print_variable(name, values, initial_value,
    ...                         transitions)
    >>> result == expected
    True
    """
    header = []
    variable_decl = ("VAR {0} : {1};".format(name, values))
    header.append(variable_decl)
    variable_init = "ASSIGN init({0}) := {1};".format(name, initial_value)
    header.append(variable_init)
    
    cases = []
    cases.append("next({0}) :=".format(name))
    cases.append("case")
    cases += transitions
    cases.append("esac;")

    return header, cases


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
    >>> expected = "next(s)=1: p_C;"
    >>> result == expected
    True
    """
    chroma_array = oracle.feature[state]
    chord = chroma.to_chord(chroma_array)
    root_name = make_root_name(chord)
    return "next({0})={1}: {2};".format(nuxmv_state_name, state,
                                        root_name)

def print_pitches(oracle, nuxmv_state_name='s'):
    """Return full nuXmv variable declaration for pitch in `oracle`.

    ----
    >>> import copy
    >>> import music21
    >>> import vmo.utils.music21_interface as vmusic
    >>> c1 = music21.chord.Chord(['C4', 'E4', 'G4'], quarterLength=1)
    >>> c2 = music21.chord.Chord(['E4', 'G4', 'B4'], quarterLength=1)
    >>> s = music21.stream.Stream([c1])
    >>> s.append(c2)
    >>> s.append(copy.deepcopy(c1))
    >>> s.append(music21.chord.Chord([], quarterLength=1))  # silent frame
    >>> oracle = vmusic.from_stream(s)
    >>> result = print_pitches(oracle)
    >>> expected = (["VAR pitchRoot : " +
    ...              "{p_Init, p_Silence, p_C, p_C#, p_D, p_E-, p_E, " +
    ...              "p_F, p_F#, p_G, p_G#, p_A, p_B-, p_B};",
    ...              "ASSIGN init(pitchRoot) := p_Init;"],
    ...             ["next(pitchRoot) :=",
    ...              "case",
    ...              "next(s)=0: p_Init;",
    ...              "next(s)=1: p_C;",
    ...              "next(s)=2: p_E;",
    ...              "next(s)=3: p_C;",
    ...              "next(s)=4: p_Silence;",
    ...              "esac;"])
    >>> result == expected
    True
    """
    name = 'pitchRoot'
    values = ("{p_Init, p_Silence, p_C, p_C#, p_D, p_E-, p_E, " +
              "p_F, p_F#, p_G, p_G#, p_A, p_B-, p_B}")
    initial_value = 'p_Init'
    
    transitions = []
    transitions.append(
        "next({})=0: p_Init;".format(nuxmv_state_name))
    for s in range(1, oracle.n_states):
        transitions.append(print_pitch_state(oracle, s, nuxmv_state_name))

    return print_variable(name, values, initial_value, transitions)

def print_pitchspace_state(oracle, original_stream, state, nuxmv_state_name='s'):
    """Return a nuXmv assignation case for pitch space for `state`.

    Keyword arguments:
        oracle: vmo.VMO.oracle
            The oracle to model.
        state: int
            The state in the oracle to print.
        original_stream: music21.stream.Stream
            The music21 stream object used to create `oracle`.
        nuxmv_state_name: str
            The name of the state in the model representing the oracle state
            (default 's').
    ----
    >>> import copy
    >>> import music21
    >>> import vmo.utils.music21_interface as vmusic
    >>> c1 = music21.chord.Chord(['C4', 'E4', 'G4'], quarterLength=1)
    >>> c2 = music21.chord.Chord(['E4', 'G4', 'B4'], quarterLength=1)
    >>> s = music21.stream.Stream([c1])
    >>> s.append(c2)
    >>> s.append(copy.deepcopy(c1))
    >>> o = vmusic.from_stream(s, threshold=0.01)
    >>> result = print_pitchspace_state(o, s, 1)
    >>> expected = "next(s)=1: {};".format(c1.root().midi)
    >>> result == expected
    True
    >>> result = print_pitchspace_state(o, s, 2)
    >>> expected = "next(s)=2: {};".format(c2.root().midi)
    >>> expected == result 
    True
    """
    import vmo.utils.music21_interface as vmusic
    
    def get_ps_root_of_frame(state):
        """Return an int representing the pitch space position of
        the root of `state`'s content.

        Return 0 if `state` holds a silent feature.
        """
        frame = vmusic.extract_frame_oracle(original_stream, oracle, state)
        chord = mus.chord.Chord(frame.pitches)
        return chord.root().midi if chord.pitches else -1

    transition = "next({0})={1}: {2};".format(nuxmv_state_name,
                                              state,
                                              get_ps_root_of_frame(state))

    return transition

def print_pitchspaces(oracle, original_stream, nuxmv_state_name='s',
                      nuxmv_pitchspace_name='pitchSpace'):
    """Return full nuXmv variable declaration for pitch space in `oracle`.

    Three possible values for variable `'pitchMotion'`:
        `'m_asc'` if melodic motion from previous frame to current is ascendant
        `'m_desc'` if melodic motion descendant
        `'m_none`' at initialization and when encountering silent frames
        
    Keyword arguments:
        oracle: vmo.VMO.oracle
            The oracle to model.
        original_stream: music21.stream.Stream
            The music21 stream object used to create `oracle`.
        nuxmv_state_name: str
            The name of the state in the model representing the oracle state
            (default 's').
    """
    states = range(1, oracle.n_states)
    local_print_motion_state = (lambda state: print_pitchspace_state(
        oracle, original_stream, state, nuxmv_state_name))
    transitions = map(local_print_motion_state, states)
    transitions.append("TRUE: -1;")  # To make the case-disjunt exhaustive. 
    
    return print_variable(nuxmv_pitchspace_name, '-1 .. 127',
                          '-1', transitions)

def print_motions_generic_define(nuxmv_motion_name='pitchMotion',
                                 nuxmv_pitchspace_name='pitchSpace'):
    """Return the generic variable declaration for melodic motion.
    
    The nuXmv values of this variable depend only on the pitch space variable.
    """                             
    output = (
        "VAR {0} : {{m_none, m_desc, m_asc}};\n".format(nuxmv_motion_name) +
        "ASSIGN init({}) := m_none;\n".format(nuxmv_motion_name) +
        "\tnext({}) :=\n".format(nuxmv_motion_name) +
        "\tcase\n" +
        "\tnext({0})<0 | {0}<0: m_none;\n".format(nuxmv_pitchspace_name) +
        "\tnext({0})>={0}: m_asc;\n".format(nuxmv_pitchspace_name) +
        "\tTRUE: m_desc;\n" +
        "\tesac;\n"        
    )
    return output
    
"""Print chromagram oracle"""

def print_oracle(oracle, enable_motions=False, include_rsfx=False,
                 init_state=None, nuxmv_state_name='s', original_stream=None,
                 nuxmv_motion_name='pitchMotion',
                 nuxmv_pitchspace_name='pitchSpace'):
    """Return a string describing `oracle`, with oracle states and pitches.

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
    >>> result = print_oracle(o, enable_motions=False)
    >>> expected = (
    ...     "MODULE main()\\n" +
    ...     "VAR s: 0..2;\\n" +
    ...     "ASSIGN init(s) := 0;\\n" +
    ...     "\\tnext(s) :=\\n" +
    ...     "\\tcase\\n" +
    ...     "\\ts=0: {1, 2};\\n" +
    ...     "\\ts=1: {1, 2};\\n" +
    ...     "\\ts=2: {1, 2};\\n" +
    ...     "\\tesac;\\n\\n" +
    ...     "VAR pitchRoot : {p_Init, p_Silence, p_C, p_C#, p_D, p_E-, p_E, " +
    ...     "p_F, p_F#, p_G, p_G#, p_A, p_B-, p_B};\\n" +
    ...     "ASSIGN init(pitchRoot) := p_Init;\\n" +
    ...     "\\tnext(pitchRoot) :=\\n" +
    ...     "\\tcase\\n" +
    ...     "\\tnext(s)=0: p_Init;\\n" +
    ...     "\\tnext(s)=1: p_C;\\n" +
    ...     "\\tnext(s)=2: p_E;\\n" +
    ...     "\\tesac;\\n\\n" +
    ...     "-- Motions not enabled."
    ...    )
    >>> result == expected
    True
    """
    if enable_motions and original_stream is None:
        raise ValueError("Must provide original stream if motions are required")
    
    if init_state is None:
        init_state = oracle.initial_state
    
    adj_lists = van.graph_adjacency_lists(oracle, include_rsfx,
                                          compress_to_forward_links=True)
    base_model = print_module(adj_lists, nuxmv_state_name=nuxmv_state_name,
                              init_state=init_state)

    def full_variable_printer(fun_output):
        header, cases = fun_output
        header_str = indent_join_lines(header, tabulate_by=0)
        cases_str = indent_join_lines(cases)
        
        var_str = "".join([header_str, cases_str])
        return var_str

    pitches_str = full_variable_printer(print_pitches(oracle, nuxmv_state_name))
    if enable_motions:
        motions_str = full_variable_printer(
            print_pitchspaces(oracle,
                              original_stream=original_stream,
                              nuxmv_state_name=nuxmv_state_name,
                              nuxmv_pitchspace_name=nuxmv_pitchspace_name)
            ) + print_motions_generic_define(
                nuxmv_pitchspace_name=nuxmv_pitchspace_name,
                nuxmv_motion_name=nuxmv_motion_name)
    else:
        motions_str = "-- Motions not enabled."
        
    return "\n".join([base_model, pitches_str, motions_str])

        
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
    return "".join(tabbed_lines)

def make_root_name(chord, nuxmv_silence_name='p_Silence'):
    """Generic function to return a valid root name for a model.

    Keyword arguments:
        chord: music21.chord.Chord input type
            The note or chord for which to extract a root.
    """
    if isinstance(chord, basestring):
        chord = [chord] 
    chord = mus.chord.Chord(chord)
    if not chord.pitches:
        return nuxmv_silence_name
    else:
        root_name = ('p_' + chord.root().name)
        return root_name

            
if __name__ == "__main__":
    import doctest
    doctest.testmod()
