"""
distances/tonnetz.py
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

"""Tonnetz distance between chords.

This module defines a distance on chromagram vectors
(12 bin pitch-class arrays).
This distance is computed by first projecting both vectors into
a 6-dimensional space replicating the circle of pure fifth and both
circles of minor and major thirds,
then computing the euclidian distance between those
two (normalized) vectors.

See: Harte and Sandler and Gasser, Detecting harmonic
     change in musical audio,
     In Proceedings of Audio and Music Computing for Multimedia Workshop,
     2006
"""

# Constants

"""Radiuses of different circles (defined such to reflect the
tonal distance through the euclidian distance on tonnetz vectors)
"""

r_fifth = 1.
r_minor_thirds = 1.
r_major_thirds = 0.5


def _make_tonnetz_matrix():
    """Return the tonnetz projection matrix."""
    pi = np.pi
    chroma = np.arange(12)

    # Define each row of the transform matrix
    fifth_x = r_fifth*(np.sin((7*pi/6) * chroma))
    fifth_y = r_fifth*(np.cos((7*pi/6) * chroma))
    minor_third_x = r_minor_thirds*(np.sin(3*pi/2 * chroma))
    minor_third_y = r_minor_thirds*(np.cos(3*pi/2 * chroma))
    major_third_x = r_major_thirds*(np.sin(2*pi/3 * chroma))
    major_third_y = r_major_thirds*(np.cos(2*pi/3 * chroma))

    # Return the tonnetz matrix
    return np.vstack((fifth_x, fifth_y,
                      minor_third_x, minor_third_y,
                      major_third_x, major_third_y))

# Define a global constant to avoid recomputations of the Tonnetz matrix
__TONNETZ_MATRIX = _make_tonnetz_matrix()


def _to_tonnetz(chromagram):
    """Project a chromagram on the tonnetz.

    Returned value is normalized to prevent numerical instabilities.  
    """
    if np.sum(np.abs(chromagram)) == 0.:
        # The input is an empty chord, return zero. 
        return np.zeros(6)
    
    _tonnetz = np.dot(__TONNETZ_MATRIX, chromagram)
    one_norm = np.sum(np.abs(_tonnetz))  # Non-zero value
    _tonnetz = _tonnetz / float(one_norm) # Normalize tonnetz vector
    return _tonnetz


def distance(a, b):
    """Compute tonnetz-distance between two chromagrams.
    
    ----
    >>> C = np.zeros(12)
    >>> C[0] = 1
    >>> D = np.zeros(12)
    >>> D[2] = 1
    >>> G = np.zeros(12)
    >>> G[7] = 1

    The distance is zero on equivalent chords
    >>> distance(C, C) == 0
    True

    The distance is symetric
    >>> distance(C, D) == distance(D, C)
    True

    >>> distance(C, D) > 0
    True
    >>> distance(C, G) < distance(C, D)
    True
    """
    [a_tonnetz, b_tonnetz] = [_to_tonnetz(x) for x in [a, b]]
    return np.linalg.norm(b_tonnetz - a_tonnetz)
