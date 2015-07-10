"""
utils/nuxmv/check.py
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

import vmo.utils.nuxmv.model as model
import vmo.utils.nuxmv.properties as props
import vmo.utils.nuxmv.parse as parser

"""Functions to call nuXmv on a model and property and return the output.""" 

# Assumes the user has installed nuxmv as a shell command
PATH_TO_NUXMV = "/home/theis/Documents/nuxmv/nuxmv-1.0.1-linux-x86_64/nuXmv"
# distutils.spawn.find_executable('nuXmv')    

def _write_model(model_str):
    """Return a new, opened, uniquely-named file descriptor to the input model.

    Warning: please close the returned file descriptor after using it.
    The written file on disk is temporary and will be deleted as soon as the
    associated file descriptor is closed.
     
    Keyword arguments:
        model_str: string
            The model to write, in nuXmv syntax
    """
    model_description = tempfile.NamedTemporaryFile('w+')
    model_description.write(model_str)
    model_description.flush()
    
    return model_description

# def _build_model(model_str):
#     """Build the model with nuXmv and write the associated files to disk.

#     Return <model_built>, the prefix (extension-free filename) of the nuXmv-built
#     model-description:
#         <model_built>.sta describes the states
#         <model_built>.tra describes the transitions

#     Keyword arguments:
#         model_str: string
#             The model to build, in nuXmv syntax 
#     """
#     model_description = _write_model(model_str)
#     model_name = os.path.splitext(model_description.name)[0]

#     subprocess32.call([PATH_TO_NUXMV,
#                        model_description.name,
#                        '-exportmodel', model_name+'.tra,sta'])
#     model_description.close()
#     return model_name

def _check_property_full(model_str, prop):
    """Check property prop on the model described by model_description

    Return nuXmv's full output as a string.
    Keyword arguments:
        model_str: string
            The model to use, in nuXmv-syntax
        prop: string
            The property to check on the model, in nuXmv syntax
    ----
    >>> import vmo.VMO.oracle as oracle
    >>> import vmo.analysis as analysis
    >>> import vmo.utils.nuxmv.parse as parser
    
    >>> o = oracle.create_oracle('f')
    >>> o.add_state(1) # create_oracle generates state 0
    >>> o.add_state(2)
    >>> adj = analysis.graph_adjacency_lists(o)
    >>> model_str = model.print_module(adj)

    Check reachability of state 2
    >>> nuxmv_output = _check_property_full(model_str,
    ...                                     "(E [s=0 U (E [s=1 U s=0])])")

    Any oracle is strongly connected (because of the forward transitions) 
    >>> parser.is_accepting(nuxmv_output)
    True
    """
    model = _write_model(model_str + "\n\nCTLSPEC {};\n".format(prop))
    
    output = subprocess32.check_output(
        [PATH_TO_NUXMV, # call nuXmv
         model.name])   # Input the model and property
    model.close()
    return output

def check_property(model_str, prop):
    nuxmv_output = _check_property_full(model_str, prop)
    return parser.is_accepting(nuxmv_output)

# def generate_path()

if __name__ == "__main__":
    import doctest
    doctest.testmod()
