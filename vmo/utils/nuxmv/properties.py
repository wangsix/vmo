"""
utils/nuxmv/properties.py
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
import copy

import music21 as music

import vmo.utils.nuxmv.model as model

"""nuXmv properties generation module."""

def pitch_equal(pitch, silence_equivalence=False, allow_init=False,
                nuxmv_pitch_name='pitchRoot',
                nuxmv_silence_name='p_Silence', nuxmv_empty_name='p_Init'):
    """Return a nuxmv string stating the current pitch root is `pitch`.
    
    Keyword arguments:
        pitch: string
            The name of the pitch to test equality with.
        silence_equivalence: bool, optional
            Whether silence should be considered equivalent to any given pitch
            (default False, since we then generate less interesting paths)
        allow_init: bool, optional
            Whether an empty state should be considered equivalent
            to any given pitch
            (default False, since we then generate less interesting paths)
        nuxmv_pitch_name: string, optional
            The name of the variable in the nuXmv model specifying
            the current pitch root (default 'pitchRoot')
        nuxmv_silence_name: string, optional
            The name of the nuxmv value specifying that the current
            state holds no notes (default 'p_Silence')
        nuxmv_empty_name: string, optional
            The name of the nuxmv value specifying that the current
            state is the initial state (default 'p_Init')
            (Note: could test for `state == 0`)
    """
    equality_test = "{0}={1}".format(nuxmv_pitch_name,
                                     pitch)
    if pitch != nuxmv_silence_name and silence_equivalence:
        equality_test = "{0} | {1}={2}".format(equality_test,
                                               nuxmv_pitch_name,
                                               nuxmv_silence_name)
    if allow_init:
        equality_test = "{0} | {1}={2}".format(equality_test,
                                               nuxmv_pitch_name,
                                               nuxmv_empty_name)
    return equality_test

def make_chord_progression(progression, exists=True,
                           silence_equivalence=False, allow_init=False,
                           nuxmv_pitch_name='pitchRoot',
                           nuxmv_silence_name='p_Silence'):
    # TODO: Fix this, the properties for fixed ranges are wrong,
    # should use a combination of EBF and EBG, because E [ BU ] actually
    # does not enforce truth on all states before m, so the strict progression
    # only states that the last state satisfies the requirement
    """Return a string stating the existence of a path following `progression`.

    Keyword arguments:
        progression: ((string, (int, int)) or (string, int) or string) sequence
            The chord progression to test for.
            Each pair in the sequence consists of:
                The name of the root note, e.g. 'C#' or 'D-'
                The quarter-length duration for which the note should be held,
                as an interval of acceptable durations.
                    A single int means an exact duration is expected.
                    A value of zero means the duration can be arbitrary.
                    If no duration is given, a value of zero is assumed.
        exists: bool
            The truth value to test
            (default True, should be False for counter-example generation). 
        silence_equivalence: bool, optional
            Whether silence should be considered equivalent to any given pitch
            (default False, since we then generate less interesting paths).
        allow_init: bool, optional
            Whether an empty state should be considered equivalent
            to any given pitch
            (default False, since we then generate less interesting paths).
        nuxmv_pitch_name: string
            The name of the nuxmv model variable representing the current pitch
            class.
        nuxmv_silence_name: string, optional
            The name of the nuxmv value specifying that the current
            state holds no notes (default 'p_Silence').
    """
    progression_copy = copy.deepcopy(progression)
    progression_copy.reverse()
    
    def progression_aux(progression):
        if not progression:
            return 'TRUE'
        else:
            next_elem = progression.pop()
            if isinstance(next_elem, str):
                (pitch, duration) = (next_elem, 0)
            else:
                (pitch, duration) = next_elem
            root_name = model.make_root_name(pitch)
            next_prop = progression_aux(progression)
            equality_test = pitch_equal(
                root_name,
                silence_equivalence=silence_equivalence,
                nuxmv_pitch_name=nuxmv_pitch_name,
                nuxmv_silence_name=nuxmv_silence_name)
            if duration == 0:
                prop = "({0}) & E [({0}) U ({1})]".format(equality_test,
                                                          next_prop)
            elif isinstance(duration, int):
                # Enforce equality for `duration` steps
                interval_eq = "EBG {0} .. {1} ({2})".format(
                    0, duration - 1, equality_test)

                # Enforce reachability of next interval in exactly `duration`
                reachability_prop = "EBF {0} .. {0} ({1})".format(
                    duration, next_prop)

                # Combine interval constraint and reachability
                prop = "({0}) & ({1})".format(interval_eq, reachability_prop)
            elif isinstance(duration, tuple) and len(duration) == 2:
                min_dur, max_dur = duration

                # Enforce equality for at least `min_dur` steps
                interval_eq = "EBG {0} .. {1} ({2})".format(
                    0, min_dur - 1, equality_test)

                # Enforce reachability of next interval
                # between `min_dur` and `max_dur`
                reachability_prop = "E [({0}) BU {1} .. {2} ({3})]".format(
                    equality_test, min_dur, max_dur, next_prop)

                # Combine interval constraint and reachability
                prop = "({0}) & ({1})".format(interval_eq, reachability_prop)
                
            else:
                error_str = "Duration {} is not of the expected type".format(
                    str(duration))
                raise TypeError(error_str +
                                " : no duration, int or int pair.")
            
            return prop
    
    return "CTLSPEC {1}(EF ({0}))".format(progression_aux(progression_copy),
                                          '' if exists else '!')
