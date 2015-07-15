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

def pitch_equal(pitch, nuxmv_pitch_name='pitchRoot',
                nuxmv_silence_name='p_Silence'):
    # if pitch == nuxmv_silence_name:
    equality_test = "{0}={1}".format(nuxmv_pitch_name,
                                     pitch)
    # else:
    #     equality_test = "{0}={1} | {0}={2}".format(nuxmv_pitch_name,
    #                                                pitch,
    #                                                nuxmv_silence_name)
    return equality_test

def make_cadence(cadence, exists=True, nuxmv_pitch_name='pitchRoot',
                 nuxmv_silence_name='p_Silence'):
    """Return a CTL string stating the existence of a path following `cadence`.

    Keyword arguments:
        cadence: (string, int) sequence
            The cadence to test for.
            Each pair in the sequence consists of:
                The name of the root note, e.g. 'C#' or 'D-'
                The quarter-length duration for which the note should be held.
                    A value of zero means the duration can be arbitrary. 
        nuxmv_pitch_name: string
            The name of the nuxmv model variable representing the current pitch
            class.
        exists: bool
            The truth value to test
            (default True, should be False for counter-example generation).
    """
    cadence_copy = copy.deepcopy(cadence)
    cadence_copy.reverse()
    
    def cadence_aux(cadence):
        if not cadence:
            return 'TRUE'
        else:
            pitch, duration = cadence.pop()
            root_name = model.make_root_name(pitch)
            next_prop = cadence_aux(cadence)
            equality_test = pitch_equal(root_name, nuxmv_pitch_name,
                                        nuxmv_silence_name)
            if duration == 0:
                prop = "({0}) & E [({0}) U ({1})]".format(equality_test,
                                                          next_prop)
            else:
                prop = "({0}) & E [({0}) BU {2} .. {2} ({1})]".format(
                    equality_test, next_prop, duration)
            return prop
    
    return "CTLSPEC {1}(EF ({0}))".format(cadence_aux(cadence_copy),
                                          '' if exists else '!')
