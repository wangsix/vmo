"""
utils/nuxmv/parse.py
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
import xml.etree.ElementTree as ET

"""Parse nuXmv output."""

def is_accepting(nuxmv_output):
    """Return whether the input call to nuXmv was accepting.

    Look for the substring "is true" with "-- specification"
    on the same line in the nuXmv's output.
    
    Keyword arguments:
        nuxmv_output: string
            The output of a call to nuXmv
    """
    lines = nuxmv_output.split('\n')
    for line in lines:
        if "-- specification" in line:
            return ("is true" in line)
    return False

def parse_path(xml_file):
    """Return the sequence of states output by nuXmv.

    Return:
        path: sequence of dictionaries
            Describe the value for a each state at each step.

    Keyword arguments:
        xml_file: opened file descriptor
            The file containing the path in XML format as output by nuXmv 
    ----
    TODO: Rewrite this doctest and remove calls to vmo and check, make it
    more independent.
        
    >>> import vmo.VMO.oracle as oracle
    >>> import vmo.analysis as analysis
    >>> import vmo.utils.nuxmv.model as model
    >>> import vmo.utils.nuxmv.check as check
    
    >>> o = oracle.create_oracle('f')
    >>> o.add_state(1)  # create_oracle generates state 0
    >>> o.add_state(2)
    >>> adj = analysis.graph_adjacency_lists(o)
    >>> model_str = model.print_module(adj, nuxmv_state_name='s')
    
    Generate a path disproving the unreachability of state 2 then 1.
    >>> xml_path = check.generate_path(model_str,
    ...                                "CTLSPEC !(EF (s=2 & (EF s=1)))")
    >>> path = parse_path(xml_path)

    The path starts in the initial state, 0.
    >>> path[0]['s'] == 0
    True

    The path end in the desired state, 0.
    >>> path[-1]['s'] == 1
    True

    The path goes through state 2.
    >>> any(state['s'] == 2 for state in path)
    True
    """
    tree = ET.parse(xml_path_output)

    root = tree.getroot()
    path = []
    for state in root.iter('state'):
        new_state = {}
        for variable in state.iter('value'):
            name = variable.attrib['variable']
            value = int(variable.text)
            new_state.setdefault(str(name), value)
        path.append(new_state)

    return path

if __name__ == "__main__":
    import doctest
    doctest.testmod()
