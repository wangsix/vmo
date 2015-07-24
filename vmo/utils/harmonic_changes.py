"""
utils/harmonic_changes.py
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
from math import floor

from scipy.signal import find_peaks_cwt as scipeaks_cwt

import vmo.distances.tonnetz as tonnetz
import vmo.utils.chromagram as chroma

"""Harmonic changes extraction.

Implements the process described by Harte, Sandler and Gasser in 'Detecting
Harmonic Changes In Musical Audio' (In Proceedings of Audio and Music Computing
for Multimedia Workshop, 2006) to extract harmonic changes from a musical
stream.
Detect maxima of the local Tonnetz-distances between subsequent events in the
stream and label those as harmonic changes.
""" 

def _get_distances(chromagram):
    """Compute local Tonnetz-distances between subsequent chromas.

    Return distances, where distances[n] is the Tonnetz-distance between
    events at offset (n-1) and (n+1) in stream.

    Keyword arguments:
        chromagram: ndarray
            a matrix whose individual columns are chroma arrays
    """
    _, duration = chromagram.shape
    
    distances = np.zeros(duration)
    for i in range(1,duration-1):
        distances[i] = tonnetz.distance(chromagram[:,i-1],
                                        chromagram[:,i+1])
    return distances

def _find_peaks_array(array, widths=np.arange(1,2)):
    """Return the indexes of the peaks in the input array.

    Simple wrapped call to scipy's function. 
    
    Keyword arguments:
        array: ndarray
            the signal to analyze
        widths: sequence, optional
            1-D array of estimated widths of peaks in the input signal
            (default np.arange(1,2), harmonic changes supposed instantaneous)
    """
    return scipeaks_cwt(array, widths=widths)

def from_chroma(chromagram, widths=np.arange(1,2)):
    """Return the indexes of the detected harmonic changes.

    Keyword arguments:
        chromagram: ndarray
            the chromagram to analyse
        widths: sequence, optional
            1-D array of estimated widths of peaks in the input signal
            (default np.arange(1,2), harmonic changes supposed instantaneous)
    """
    distances = _get_distances(chromagram)
    peak_indexes = _find_peaks_array(distances, widths=widths)
    return peak_indexes

def from_stream_by_indexes(stream, framesize=1/8., sigma=None,
                           widths=np.arange(1,2)):
    """Return indexes of harmonic changes in the input stream.

    Compute the stream's chromagram and call from_chroma on it.
     
    Keyword arguments:
        stream: music21.stream.Stream
            the stream to analyze
        framesize: float, optional
            the size of analysis frames to use, lower = more precise, 
            in quarter length (default 1/16.)
        sigma: float, optional
            the smoothing parameter used to compute the chromagram
            (default None, value is then set within the chromagram module) 
        widths: sequence
            1-D array of estimated widths of peaks in the input signal
            (default np.arange(1,2), harmonic changes supposed instantaneous)
    ----
    >>> cM = music21.chord.Chord(['C4', 'E4', 'G4'], quarterLength=4.)
    >>> cMo = music21.chord.Chord(['C4', 'E4', 'G4', 'C5'], quarterLength=4.)
    >>> eM = music21.chord.Chord(['E4', 'G#4', 'B4'], quarterLength=4.)
    >>> dm = music21.chord.Chord(['D4', 'F4', 'A4'], quarterLength=4.)

    >>> s_chords = music21.stream.Stream([cM])
    >>> for chord in [cMo, eM, dm]:
    ...     s_chords.append(chord)

    >>> from_stream_by_indexes(s_chords, framesize=1/8.)
    [64, 96]
    """ 
    chromagram = chroma.from_stream(stream, framesize=framesize,
                                    smooth=True, sigma=sigma)
    peak_indexes = from_chroma(chromagram, widths=widths)
    return peak_indexes

def from_stream_by_offsets(stream, framesize=1/8., sigma=None,
                           widths=np.arange(1,2)):
    """Return quarter-length offsets of harmonic changes in the input stream.

    Compute the stream's chromagram and call from_chroma on it.
     
    Keyword arguments:
        stream: music21.stream.Stream
            the stream to analyze
        framesize: float, optional
            the size of analysis frames to use, lower = more precise, 
            in quarter length (default 1/16.)
        sigma: float, optional
            the smoothing parameter used to compute the chromagram
            (default None, value is then set within the chromagram module) 
        widths: sequence
            1-D array of estimated widths of peaks in the input signal
            (default np.arange(1,2), harmonic changes supposed instantaneous)
    ----
    >>> cM = music21.chord.Chord(['C4', 'E4', 'G4'], quarterLength=4.)
    >>> cMo = music21.chord.Chord(['C4', 'E4', 'G4', 'C5'], quarterLength=4.)
    >>> eM = music21.chord.Chord(['E4', 'G#4', 'B4'], quarterLength=4.)
    >>> dm = music21.chord.Chord(['D4', 'F4', 'A4'], quarterLength=4.)

    >>> s_chords = music21.stream.Stream([cM])
    >>> for chord in [cMo, eM, dm]:
    ...     s_chords.append(chord)
    
    >>> harmonic_changes_fs8 = from_stream_by_offsets(s_chords, framesize=1/8.)
    >>> harmonic_changes_fs8
    [8.0, 12.0]
    
    >>> harmonic_changes_fs8 == from_stream_by_offsets(s_chords, framesize=1/16.)
    True
    """ 
    peak_indexes = from_stream_by_indexes(stream, framesize=framesize,
                                          sigma=sigma, widths=widths)
    return [x*framesize for x in peak_indexes]
    
if __name__ == "__main__":
    import doctest
    import music21
    doctest.testmod()
