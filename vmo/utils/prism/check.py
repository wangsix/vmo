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
import subprocess32
import distutils.spawn
import os.path

import vmo.utils.prism.model as model
import vmo.utils.prism.properties as props

"""Functions to call PRISM on a model and property and return the output.""" 

# Assumes the user has installed prism as a shell command
PATH_TO_PRISM = distutils.spawn.find_executable('prism')    

def _write_model(model_str):
    """Return a new, opened, uniquely-named file descriptor to the input model.

    Warning: please close the returned file descriptor after using it.
    The written file on disk is temporary and will be deleted as soon as the
    associated file descriptor is closed.
     
    Keyword arguments:
        model_str: string
            The model to write, in PRISM syntax
    """
    model_description = tempfile.NamedTemporaryFile('w+')
    model_description.write(model_str)
    model_description.flush()
    
    return model_description

def _build_model(model_str):
    """Build the model with PRISM and write the associated files to disk.

    Return <model_built>, the prefix (extension-free filename) of the PRISM-built
    model-description:
        <model_built>.sta describes the states
        <model_built>.tra describes the transitions

    Keyword arguments:
        model_str: string
            The model to build, in PRISM syntax 
    """
    model_description = _write_model(model_str)
    model_name = os.path.splitext(model_description.name)[0]

    subprocess32.call([PATH_TO_PRISM,
                       model_description.name,
                       '-exportmodel', model_name+'.tra,sta'])
    model_description.close()
    return model_name

def check_property(model_str, prop):
    """Check property prop on the model described by model_description

    Return PRISM's full output as a string.
    Keyword arguments:
        model_str: string
            The model to use, in PRISM-syntax
        prop: string
            The property to check on the model, in PRISM syntax
    ----
    >>> import vmo.VMO.oracle as oracle
    >>> import vmo.analysis as analysis
    >>> import vmo.utils.probabilities as probs
    >>> import vmo.utils.prism.parse as parser
    
    >>> o = oracle.create_oracle('f')
    >>> o.add_state(1) # create_oracle generates state 0
    >>> o.add_state(2)
    >>> adj = analysis.graph_adjacency_lists(o)
    >>> prob_adj = probs.uniform(adj)
    >>> model_str = model.print_dtmc_module(prob_adj)

    Check reachability of state 2
    >>> prism_output = check_property(model_str, "P>0 [ F s=2]")

    Any oracle is strongly connected (because of the forward transitions) 
    >>> parser.is_accepting(prism_output)
    True
    """
    model = _write_model(model_str)
    
    output = subprocess32.check_output(
        [PATH_TO_PRISM, # call prism
         model.name,    # Input the model
         '-dtmc',       # Read the model as a DTMC
         '-pf', prop])  # Input the property to check
    model.close()
    return output

if __name__ == "__main__":
    import doctest
    doctest.testmod()
