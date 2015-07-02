"""
utils/prism/check.py
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
from time import strftime
import tempfile
import subprocess32 as subp

import vmo.utils.prism.model as model
import vmo.utils.prism.properties as props

"""Functions to call PRISM on a model and property and return the output.""" 

def _write_model(model_str):
    """Return a new, opened, uniquely-named file descriptor to the input model.
    
    Warning: be careful to close the output file descriptor after using it
    Keyword arguments:
        model_str: string
            The PRISM model to write
    """
    # Generate a unique identifier
    time = strftime("%Y-%m-%d__%H_%M_%S")
    model_name = "model-{0}".format(time)

    # Build the model into file <filename>
    model_description = tempfile.NamedTemporaryFile('w+')
    model_description.write(model_str)
    model_description.flush()
    
    return model_description

def _build_model(model_str):
    """Build the model with PRISM and return the extension-free filename.
    
    Warning: be careful to close the output file descriptor after using it
    Keyword arguments:
        model_str: string
            The PRISM model to write
    """
    model_description = _write_model(model_str)
    
    subp.call(['/home/theis/Documents/prism/prism-4.2.1-src/bin/prism',
               model_description.name,
               '-exportmodel', model_name+'.tra,sta'])
    model_description.close()
    return model_name

def check_property(model_str, prop):
    """Check property prop on the model described by model_description
    
    ----
    >>> o = oracle.create_oracle('f')
    >>> o.add_state(0)
    >>> o.add_state(1)
    >>> o.add_state(2)
    >>> adj = analysis.graph_adjacency_lists(o)
    >>> prob_adj = probs.uniform(adj)
    >>> model_str = model.print_dtmc_module(prob_adj)
    >>> check_property(model_name, "P>0 [ F s=2]")
    TODO : [expect something]
    """
    model = _write_model(model_str)
    # fd.seek(0)
    # print fd.readlines()
    # URGENT TODO: make prism path platform independant!
    output = subp.check_output(['/home/theis/Documents/prism/prism-4.2.1-src/bin/prism',
                                model.name,
                                '-dtmc',
                                '-pf', prop])
    model.close()
    return output
