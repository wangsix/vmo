"""
utils/nuxmv/check.py
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
from time import strftime
import tempfile
import subprocess32
from distutils.spawn import find_executable
import os.path

import vmo.utils.nuxmv.model as model
import vmo.utils.nuxmv.properties as props
import vmo.utils.nuxmv.parse as parser

"""Functions to call nuXmv on a model and property and return the output.""" 

# Assumes the user has installed nuxmv as a shell command

def get_nuxmv_path():
    path = max(find_executable('nuXmv'),
               find_executable('nuxmv'))
    if path is None:
        raise Exception("No command 'nuXmv' or 'nuxmv'" +
                        "is available in the prompt")
    return path

PATH_TO_NUXMV = get_nuxmv_path()


def _write_model(model_str):
    """Return a new, opened, uniquely-named file descriptor to the input model.

    Warning: please close the returned file descriptor after using it.
    The written file on disk is temporary and will be deleted as soon as the
    associated file descriptor is closed.
     
    Keyword arguments:
        model_str: string
            The model to write, in nuXmv syntax
    """
    model_description = tempfile.NamedTemporaryFile('w+', suffix='.smv')
    model_description.write(model_str)
    model_description.flush()
    
    return model_description

def _write_model_and_property(model, properties):
    """Return a new, opened, uniquely-named file descriptor to the input model.

    Warning: please close the returned file descriptor after using it.
    The written file on disk is temporary and will be deleted as soon as the
    associated file descriptor is closed.
     
    Keyword arguments:
        model_str: string
            The model to write, in nuXmv syntax
        prop: list of string
            The property (or properties) to check, in nuXmv syntax 
    """
    properties_str = ";\n".join(properties)
    return _write_model(model + '\n\n' + properties_str + ';\n')

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
    ...                                     "CTLSPEC (EF s=2)")

    Any oracle is strongly connected (because of the forward transitions) 
    >>> parser.is_accepting(nuxmv_output)
    True
    """
    model = _write_model(model_str)

    model.write("\n\n" + prop + ";\n")
    model.flush()
            
    output = subprocess32.check_output(
        [PATH_TO_NUXMV, # call nuXmv
         model.name])   # Input the model and property
    model.close()
    return output

def check_property(model_str, prop):
    """Return the truth value of `prop` on the model described by `model_str`

    Keyword arguments:
        model_str: string
            The model to work on, in nuXmv syntax
        prop: string
            The property to check, in nuXmv syntax
    """
    nuxmv_output = _check_property_full(model_str, prop)
    return parser.is_accepting(nuxmv_output)

def make_counterexample(model_str, prop):
    """Return a sequence of states in the model disproving prop.

    Each state in the sequence is a dictionary, with keys the states
    in the model.

    Return None if `prop` is true on the given model.

    Keyword arguments:
        model_str: string
            The model to work on, in nuXmv syntax
        prop: string
            The property for which to exhibit a counter-example,
            in nuXmv syntax
    ----
    >>> import vmo.VMO.oracle as oracle
    >>> import vmo.analysis as analysis
    >>> import vmo.utils.nuxmv.parse as parser

    >>> o = oracle.create_oracle('f')
    >>> o.add_state(1)  # create_oracle generates state 0
    >>> o.add_state(2)
    >>> adj = analysis.graph_adjacency_lists(o)
    >>> model_str = model.print_module(adj)

    Fail to generate a path disproving the reachability of state 2.
    >>> failure = make_counterexample(model_str,
    ...                               "CTLSPEC (E [s=0 U (E [s=1 U s=0])])")
    >>> failure is None
    True
    
    Generate a path disproving the unreachability of state 2 then 1.
    >>> path = make_counterexample(model_str,
    ...                            "CTLSPEC !(EF (s=2 & (EF s=1)))")
    >>> path is not None
    True

    The generated path starts in the initial state, 0.
    >>> int(path[0]['s']) == 0
    True
    
    The generated path ends in the requested state, 1.
    >>> int(path[-1]['s']) == 1
    True

    The generated path goes through state 2 at some time.
    >>> any(int(state['s']) == 2 for state in path)
    True
    """
    model = _write_model_and_property(model_str, [prop])
    commands = tempfile.NamedTemporaryFile('w+', suffix='.xmv')
    path_xml = tempfile.NamedTemporaryFile('w+', suffix='.xml')
    nuxmv_output = tempfile.TemporaryFile('w+', suffix='.txt')
    nuxmv_errors = tempfile.TemporaryFile('w+', suffix='.txt')
    
    commands_str = ("read_model -i \"{0}\"\n".format(model.name) +
                    "flatten_hierarchy\n" +
                    "encode_variables\n" +
                    "build_model\n" +
                    "check_ctlspec\n" +
                    "set default_trace_plugin 4\n" +
                    "show_traces -o {0}\n".format(path_xml.name) +
                    "quit\n")
    commands.write(commands_str)
    commands.flush()

    with open(os.devnull, 'w') as NULL:
        subprocess32.call([PATH_TO_NUXMV,  # Call nuXmv
                           '-source', commands.name],  # Execute the commands
                           stdout=nuxmv_output,  # Redirect command-line output
                           stderr=NULL) 
    nuxmv_output.flush()
    nuxmv_output.seek(0)
    model.close()
    commands.close()

    if "is true" in nuxmv_output.read():
        # prop is true on the model
        path_xml.close()
        nuxmv_output.close()
        return None

    nuxmv_output.close()
    
    path = parser.parse_path(path_xml)
    path_xml.close()
    return path
        
if __name__ == "__main__":
    import doctest
    doctest.testmod()

