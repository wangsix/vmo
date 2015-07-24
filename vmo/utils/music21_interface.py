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
import vmo.utils.chromagram as vchroma

"""Oracle related functions."""

def from_stream(stream, framesize=1.0, threshold=0,
                suffix_method='inc', weights=None,
                dfunc='tonnetz', dfunc_handle=None):
    chroma = vchroma.from_stream(stream, framesize=framesize)
    features = np.array(chroma).T
    oracle = voracle.build_oracle(features, 'v', feature='chromagram',
                          threshold=threshold, suffix_method=suffix_method,
                          weights=weights,
                          dfunc=dfunc, dfunc_handle=dfunc_handle)
    oracle.framesize = framesize  # store `framesize` information in object
    return oracle

def oracle_from_corpus(name, **kwags):
    # Unstable, can break depending on the music21 structure of the piece
    # extracted from the corpus (e.g. if it's grouped in staves), use smart
    stream = mus.corpus.parse(name).flat.notes
    oracle = vmusic.from_stream(stream, **kwags)
    return oracle


"""Generation functions."""

    
def morph_streams(query, target, framesize=1.0, threshold=0,
                  suffix_method='inc', weights=None, dfunc='tonnetz',
                  dfunc_handle=None):
    """Apply `query`'s chord progression to `target`.

    Return:
        path: int list
            A path on `target` following the extracted chord progression
            if such a path exists, given in terms of frame indexes
    Keyword arguments:
        query: music21.stream.Stream
            The piece from which to extract the chord preogression
        target: music21.stream.Stream
            The material with which to generate music.
    """
    import vmo.logics.model_checking as vmodel
    
    chord_progression = get_chord_progression(query)
    oracle = vmusic.from_stream(target)
    path = vmodel.make_piecewise_chord_progression(oracle, [chord_progression])
    return path

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


"""Stream analysis"""

def get_chord_progression(stream, framesize=1.0, overlap=0.0,
                          smooth=False, sigma=None):
    chromagram = vchroma.from_stream(stream, framesize=framesize,
                                     overlap=overlap, smooth=smooth,
                                     sigma=sigma)
    chromagram = np.array(chromagram).T
    chords = map(vchroma.to_chord, chromagram)
    def get_name(chord):
        if not chord.pitches:
            return 'silence'
        else:
            return chord.root().name
    root_names = map(get_name, chords)
    return root_names

def progression_from_tonic(tonic, progression='authentic', mode='major'):
    """Return the pitches for the chosen `progression` with given `tonic`.
    
    Keyword arguments:
        tonic: string
            The first degree of the scale in which to generate the progression.
        progression: string or int list, optional
            The type of chord progression, supported parameters are:
                'authentic': V-I cadence
                'plagal': IV-I cadence
                a list of degrees for an arbitrary progression
                (default 'authentic').
        mode: string, optional
            The mode in which to generate the chord progression
            (default 'major').
    ----
    >>> progression_from_tonic('C', 'authentic')
    ['G', 'C']
    >>> progression_from_tonic('E', 'plagal')
    ['A', 'E']
    >>> progression_from_tonic('D', [8, 4, 5, 1])
    ['D', 'G', 'A', 'D']
    """
    p_tonic = mus.note.Note(tonic).pitchClass
    def make_degree(degree):
        degree = ((degree - 1) % 7) + 1  # degree is now in range 1 .. 7
        
        if degree == 1:
            return p_tonic
        elif degree == 2:
            return (p_tonic + 2) % 12
        elif degree == 3:
            if mode == 'major':
                return (p_tonic + 4) % 12
            elif mode == 'minor':
                return (p_tonic + 3) % 12
        elif degree == 4:
            return (p_tonic + 5) % 12
        elif degree == 5:
            return (p_tonic + 7) % 12
        elif degree == 6:
            if mode == 'major':
                return (p_tonic + 9) % 12
            elif mode == 'minor':
                return (p_tonic + 8) % 12
        elif degree == 7:
            return (p_tonic + 11) % 12
        else:
            # degree is in the range 1 .. 7, so mode in unsupported
            raise ValueError("Mode {} is unsupported".format(mode))

    if isinstance(progression, basestring):
        if progression == 'authentic':
            classes = [make_degree(5), p_tonic]
        elif progression == 'plagal':
            classes = [make_degree(4), p_tonic]
        else:
            error_string = ("Cadence type {}".format(progression) +
                            "is unsupported (yet)")
            raise ValueError(error_string)
    elif isinstance(progression, list):
        classes = map(make_degree, progression)
    else:
        raise TypeError("Progression should be a string or a list of integers")

    pitches = map(mus.pitch.Pitch, classes)
    pitch_names = map(lambda pitch: pitch.name, pitches)
    return pitch_names


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


if __name__ == "__main__":
    import doctest
    doctest.testmod()
