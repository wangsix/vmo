"""
utils/music21_interface.py
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
import copy

import music21 as mus

import vmo.VMO.oracle as voracle
import vmo.distances.chromagram as vchroma

"""Oracle related functions."""

def from_stream(stream, framesize=1.0, threshold=0,
                suffix_method='inc', weights=None,
                dfunc='tonnetz', dfunc_handle=None):
    chroma = chromagram.from_stream(stream, framesize=framesize)
    features = np.array(chroma).T
    oracle = voracle.build_oracle(features, 'v', feature='chromagram',
                          threshold=threshold, suffix_method=suffix_method,
                          weights=weights,
                          dfunc=dfunc, dfunc_handle=dfunc_handle)
    oracle.framesize = framesize  # store `framesize` information in object
    return oracle

"""Generation functions."""

def path_to_stream(original, path, framesize=1.0):
    """Return a new stream from `original` following the path `offsets`.

    Keyword arguments:
        original_stream: music21.stream.Stream
            The stream on which to follow the path.
        path: int sequence
            The path given as a sequence of states within the oracle.
        framesize: float, optional
            The duration of each frame in the sequence.
    """
    new_stream = mus.stream.Stream()
    
    # Accounting for the fact that the first state of any oracle in empty. 
    offsets = [framesize * (state - 1) for state in path if state != 0]

    def insertFrame(offset, i):
        extracted = extract_frame(original, offset, framesize)
        for note in extracted.notes:
            note_copy = copy.deepcopy(note)
            note_copy.offset = i * framesize + (note.offset - offset)
            new_stream.insert(note_copy)

    for i, offset in enumerate(offsets):
        insertFrame(offset, i)
    return new_stream


"""Generic helper functions."""

def extract_frame(stream, offset_start, framesize):
    result = stream.getElementsByOffset(
        offset_start,
        offsetEnd=offset_start+framesize,
        # Don't include notes from `stream` starting in its next frame
        includeEndBoundary=False,
        # Only include notes starting in the extracted frame
        mustBeginInSpan=True
        )
    return result

def extract_frame_oracle(stream, oracle, state):
    framesize = oracle.framesize
    offset_start = framesize * (state - 1)
    return extract_frame(stream, offset_start, framesize)
