'''
harmonic_changes.py
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
'''

import numpy as np
from math import floor

from scipy.signal import find_peaks_cwt as scipeaks_cwt

import vmo.distances.tonnetz as tonnetz
import vmo.distances.chromagram as chroma

'''Harmonic changes extraction.

Implements the process described by Harte, Sandler and Gasser in 'Detecting
Harmonic Changes In Musical Audio' (In Proceedings of Audio and Music Computing
for Multimedia Workshop, 2006) to extract harmonic changes from a musical
stream.
Detect maxima of the local Tonnetz-distances between subsequent events in the
stream and label those as harmonic changes.
''' 

def _get_distances(stream, framesize=1/16.):
    '''Compute local Tonnetz-distances between subsequent events in the stream.

    Return distances, where distances[n] is the Tonnetz-distance between
    events at offset (n-1) and (n+1) in stream.

    ----
    >>> cM = music21.chord.Chord(['C4', 'E4', 'G4'], quarterLength=4.)
    >>> cMo = music21.chord.Chord(['C4', 'E4', 'G4', 'C5'], quarterLength=4.)
    >>> eM = music21.chord.Chord(['E4', 'G#4', 'B4'], quarterLength=4.)
    >>> dm = music21.chord.Chord(['D4', 'F4', 'A4'], quarterLength=4.)
    >>> s_chords = music21.stream.Stream([cM])
    >>> for chord in [cMo, eM, dm]:
    ...     s_chords.append(chord)
    >>> find_peaks(s_chords, framesize=1/8.)
    [8.0, 12.0]
    '''
    sigma = 8 # smoothing parameter, as defined in [hsg'06]
    # TODO: 8 is probably too big. Should use something related to
    #    the quarter-length window-size of the chromagram construction 
    chromagram = chroma.from_stream(stream, smooth=True, sigma=sigma,
                                    framesize=framesize)
    _, duration = chromagram.shape
    
    distances = np.zeros(duration)
    for i in range(1,duration-1):
        distances[i] = tonnetz.distance(chromagram[:,i-1],
                                        chromagram[:,i+1])
    return distances

def _find_peaks_array(array):
    '''Return the indexes of the peaks in the input array.'''
    return scipeaks_cwt(array, np.arange(1,2))

def find_peaks(stream, framesize=1.0):
    '''Return the quarter-length offsets of the detected harmonic changes.'''
    peak_indexes = _find_peaks_array(_get_distances(stream, framesize=framesize))
    return [x*framesize for x in peak_indexes]

if __name__ == "__main__":
    import doctest
    import music21
    doctest.testmod()
