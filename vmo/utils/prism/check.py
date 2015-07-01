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
import subprocess32 as subp

import vmo.utils.prism.model as model
import vmo.utils.prism.properties as props

"""Functions to call PRISM on a model and property and return the output.""" 

def write_model(model_str):
    """Return an opened file descriptor for the input model.

    Warning: be careful to close the output file descriptor after using it
    Keyword arguments:
        model_str: string
            The PRISM model to write
    """
    # Generate a unique identifier
    time = strftime("%Y-%m-%d__%H_%M_%S")
    filename = "model-{0}.pm".format(time)
    fd = open("model-{0}.pm", 'w')
    fd.write(model_str)
    return fd

def check_property(model_str, prop):
    fd = write_model(model_str)
    output = subp.check_output(['prism', fd.filename, prop])
    return output
