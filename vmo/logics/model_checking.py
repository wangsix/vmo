"""
logics/model_checking.py
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

"""Model-checking various logics (incl. CTL, LTL) on an oracle.

Currently uses nuXmv as a backend.
"""

import music21 as mus

import vmo.utils.nuxmv.model as nuxmv_model
import vmo.utils.nuxmv.check as nuxmv_check
import vmo.utils.nuxmv.properties as nuxmv_props

def _init_modules(model_checker='nuxmv'):
    """Exports the appropriate modules given a model-checking backend."""
    if model_checker == 'nuxmv':
        model = nuxmv_model
        check = nuxmv_check
        properties = nuxmv_props
    else:
        raise ArgumentError("Unupported model-checker.")
    return model, check, properties

def check_property(oracle, prop, start=None, include_rsfx=False,
                   model_checker='nuxmv'):
    """Return the truth value of `prop` on `oracle`.

    Keyword arguments:
        model_str: string
            The model to work on, in nuXmv syntax
        prop: string
            The property to check, in nuXmv syntax
        start: int, optional
            The index of the state from which the generated path should start
            (defaults to `oracle`'s initial state)
        include_rsfx: bool, optional
            Whether reverse suffix links should be included in the graph
            extracted from `oracle` (default `False`)
    """
    if start is None:
        start = oracle.initial_state
    model, check = _init_modules(model_checker)
    
    model_str = model.print_oracle(oracle, include_rsfx=include_rsfx,
                                   init_state=start)
    truth_value = check.check_property(model_str, prop)
    return truth_value

def make_counter_example(oracle, prop, start=None, include_rsfx=False,
                         model_checker='nuxmv'):
    """Return a counter-example for `prop` on `oracle`, if it exists.

    Keyword arguments:
        model_str: string
            The model to work on, in nuXmv syntax
        prop: string
            The property to disprove, in nuXmv syntax
        start: int, optional
            The index of the state from which the generated path should start
            (defaults to `oracle`'s initial state)
        include_rsfx: bool, optional
            Whether reverse suffix links should be included in the graph
            extracted from `oracle` (default `False`)
    """
    if start is None:
        start = oracle.initial_state
    model, check = _init_modules(model_checker)
    
    model_str = model.print_oracle(oracle, include_rsfx=include_rsfx,
                                   init_state=start)
    counterexample = check.make_counterexample(oracle, prop)
    return counterexample


"""Chord progressions"""

def agglomerate_progression(progression):
    """Sequence of notes with durations from a sequence of single notes.

    Return a sequence of (str, (int, int)) pairs.
    Each pair has the form (`note`, (`min_dur`, `max_dur`)), where
    `min_dur` is the number of successive occurrences of `note`
    in the original sequence. 
    ----
    >>> result = agglomerate_progression(['C', 'D', 'E', 'E',
    ...                                   'E', 'D', 'C', 'C', 'F#'])
    >>> expected = [('C', (1, None)), ('D', (1, None)), ('E', (3, None)),
    ... ('D', (1, None)), ('C', (2, None)), ('F#', (1, None))]
    >>> result == expected
    True
    """
    def aux(l, acc):
        if not l:
            note, minimum_duration = acc
            maximum_duration = None
            l.append((note, (minimum_duration, maximum_duration)))
            return l
        else:
            next_note = l[0]
            previous_note = acc[0]
            if next_note == previous_note:
                acc = (acc[0], acc[1] + 1)
                return aux(l[1:], acc)
            else:
                previous_note, minimum_duration = acc
                maximum_duration = None
                new_acc = (next_note, 1)

                result = aux(l[1:], new_acc)
                result.append((previous_note,
                               (minimum_duration, maximum_duration)))
                return result
        
    if not progression:
        return []
    else:
        acc = (progression[0], 1)
        result = aux(progression[1:], acc)
        result.reverse()
        return result
    
def make_progression(oracle, progressions, enable_motions=False, start=None,
                     original_stream=None, include_rsfx=False,
                     silence_equivalence=False, allow_init=False,
                     model_checker='nuxmv'):
    """Return a path in `oracle` reaching the given `progression` from `start`.

    Keyword arguments:
        oracle: vmo.VMO.VMO
            The oracle on which to generate a path.
        progressions: sequences of (str, (int, int))
            The piecewise chord progression to test for, of the form:
            [PROG_1, PROG_2, ..., PROG_n].
            Each PROG_i should be continously satisfied.
            Arbitrary paths can be taken to connect each PROG_i though.
            
            Each PROG_i is a sequence of pairs of the form:
                The name of the root note, e.g. 'C#' or 'D-'
                The quarter-length duration for which the note should be held,
                as an interval `(min_dur, max_dur)` of acceptable durations.
                    `min_dur` and `max_dur` can be either `int`s or `None`.

        enable_motions: bool
            Allow writing of chords as '+B' or '-F#' to specify melodic motion
            used to reached chord (default `False`).
        start: int, optional
            The index of the state from which the generated path should start
            (defaults to `oracle`'s initial state).
        include_rsfx: bool, optional
            Whether reverse suffix links should be included in the graph
            extracted from `oracle` (default `False`).
        silence_equivalence: bool, optional
            Whether silence should be considered equivalent to any given pitch
            (default False, since we then generates less uninteresting paths).
        allow_init: bool, optional
            Whether the initial state should be considered equivalent
            to any given pitch
            (default False, since we then generate less interesting paths)
    """
    if start is None:
        start = oracle.initial_state
    model, check, properties = _init_modules(model_checker)
    
    model_str = model.print_oracle(oracle, include_rsfx=include_rsfx,
                                   enable_motions=enable_motions,
                                   init_state=start,
                                   original_stream=original_stream)
    progression_prop = properties.makepprogression(
        progressions, exists=False,
        silence_equivalence=silence_equivalence,
        allow_init=allow_init)
        
    return (check.make_counterexample(model_str, progression_prop))


def make_progression_from_degrees(oracle, progressions, original_stream=None,
                                  mode='major', enable_motions=False,
                                  start=None, include_rsfx=False,
                                  silence_equivalence=False, allow_init=False,
                                  model_checker='nuxmv'):
    """Return a path in `oracle` reaching the given `progressions` from `start`.

    Lifted, degree-based version of `make_progression`:
    Test for all 12 possibilities of instantiation of the degrees following the
    choice of an arbitrary tonic.
    
    Return: (dict sequence, str)
        The first existing path and the associated tonic, cf. `make_progression`
        for the return dict's syntax.
    
    Keyword arguments:
        oracle: vmo.VMO.VMO
            The oracle on which to generate a path.
        progressions: sequences of (int, (int, int))
            The piecewise chord progression to test for, of the form:
            [PROG_1, PROG_2, ..., PROG_n].
            Each PROG_i should be continously satisfied.
            Arbitrary paths can be taken to connect each PROG_i though.
            
            Each PROG_i is a sequence of pairs of the form:
                The degree of the root note, an int between 1 and 12.
                The quarter-length duration for which the note should be held,
                as an interval `(min_dur, max_dur)` of acceptable durations.
                    `min_dur` and `max_dur` can be either `int`s or `None`.

        enable_motions: bool
            Allow writing of degrees as '+1' or '-2' to specify melodic motion
            used to reached chord (default `False`).
        mode: string, optional
            The mode in which to generate the chord progression
            (default 'major').
        start: int, optional
            The index of the start from which the generated path should start
            (defaults to `oracle`'s initial state)
        include_rsfx: bool, optional
            Whether reverse suffix links should be included in the graph
            extracted from `oracle` (default `False`)
        silence_equivalence: bool, optional
            Whether silence should be considered equivalent to any given pitch
            (default False, since we then generates less uninteresting paths)
        allow_init: bool, optional
            Whether the initial state should be considered equivalent
            to any given pitch
            (default False, since we then generate less interesting paths)
    """
    import vmo.utils.music21_interface as vmusic

    if start is None:
        start = oracle.initial_state
    model, check, properties = _init_modules(model_checker)

    result = None
    
    tonic = 0
    model_str = model.print_oracle(oracle, enable_motions=enable_motions,
                                   init_state=start,
                                   include_rsfx=include_rsfx,
                                   original_stream=original_stream)
    while result is None and tonic < 12:
        tonic_name = mus.pitch.Pitch(tonic).name
        inst_progs = (lambda progression:
            vmusic.instantiate_degree_progression(tonic_name, progression=progression,
                                          mode=mode,
                                          enable_motions=enable_motions)
                                          )
        progressions_inst = map(inst_progs, progressions)
        progression_prop = properties.make_progression(
            map(agglomerate_progression, progressions_inst),
            exists=False,
            silence_equivalence=silence_equivalence)
        tonic += 1
        result = check.make_counterexample(model_str, progression_prop)
        
    return (result, tonic_name)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
