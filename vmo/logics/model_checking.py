"""
logics/model_checking.py
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

"""Model-checking various logics (incl. CTL, LTL) on an oracle.

Currently uses nuXmv as a backend..
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
    counterexample = check.make_example(oracle, prop)
    return counterexample
    
def make_chord_progression(oracle, progression, start=None, include_rsfx=False,
                           silence_equivalence=False,
                           allow_init=False,
                           model_checker='nuxmv'):
    """Return a path in `oracle` reaching the given `progression` from `start`.

    Keyword arguments:
        oracle: vmo.VMO.VMO
            The oracle on which to generate a path.
        progression: (string, int) sequence
            The chord progression to test for.
            Each pair in the sequence consists of:
                The name of the root note, e.g. 'C#' or 'D-'
                The length for which the note should be held, in quarter length.
        start: int, optional
            The index of the state from which the generated path should start
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
    if start is None:
        start = oracle.initial_state
    model, check, properties = _init_modules(model_checker)
    
    model_str = model.print_oracle(oracle, include_rsfx=include_rsfx,
                                   init_state=start)
    progression_prop = properties.make_chord_progression(
        progression, exists=False,
        silence_equivalence=silence_equivalence,
        allow_init=allow_init)
        
    return (check.generate_path(model_str, progression_prop))
