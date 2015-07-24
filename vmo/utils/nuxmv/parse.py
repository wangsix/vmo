"""
utils/nuxmv/parse.py
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
    """
    tree = ET.parse(xml_file)

    root = tree.getroot()
    path = []
    for state in root.iter('state'):
        new_state = {}
        for variable in state.iter('value'):
            name = variable.attrib['variable']
            value = variable.text
            if name != '_PITCH_TYPE':
                # Exclude variable _PITCH_TYPE used only for type declaration
                new_state.setdefault(str(name), value)
        path.append(new_state)

    return path

if __name__ == "__main__":
    import doctest
    doctest.testmod()
