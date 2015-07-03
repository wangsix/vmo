"""
utils/prism/parse.py
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

"""Parse PRISM output."""

def is_accepting(prism_output):
    """Return whether the input call to PRISM was accepting.

    Look for the substring "Result: true" in the text output
    Keyword arguments:
        prism_output: string
            The output of a call to PRISM
    """
    # TODO: check robustness wrt PRISM output
    return "Result: true" in prism_output

def parse_path(path_output):
    """Parse a path as output by PRISM.

    Return a sequence of tuples and a tuple describing the name of each
    component in the returned tuples"""
    raise NotImplementedError("Need to know if PRISM supports path generation")
