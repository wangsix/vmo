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

Currently uses nuXmv as a backend, interface is in vmo/utils/nuxmv/.
"""

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

def check_property(oracle, prop, model_checker='nuxmv'):
    """Return the truth value of `prop` on `oracle`.

    Keyword arguments:
        model_str: string
            The model to work on, in nuXmv syntax
        prop: string
            The property to check, in nuXmv syntax
    """
    model, check = _init_modules(model_checker)
    
    model_str = model.print_oracle(oracle)
    return (check.check_property(model_str, prop))

def generate_cadence(oracle, cadence, start=None, model_checker='nuxmv'):
    """Generate a path in `oracle` reaching the given `cadence` from `start`.

    Keyword arguments:
        oracle: vmo.VMO.VMO
            The oracle on which to generate a path.
        cadence: (string, int) sequence
            The cadence to test for.
            Each pair in the sequence consists of:
                The name of the root note, e.g. 'C#' or 'D-'
                The length for which the note should be held, in quarter length.
    """
    if start is None:
        start = oracle.initial_state
    model, check, properties = _init_modules(model_checker)
    
    if model_checker == 'nuxmv':
        model_str = model.print_oracle(oracle, init_state=start)
        cadence_prop = nuxmv_props.make_cadence(cadence, False)
        
        return (check.generate_path(model_str, cadence_prop))
    else:
        raise ArgumentError("Unupported model-checker.")
