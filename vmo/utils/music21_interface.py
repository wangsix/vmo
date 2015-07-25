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
    oracle = from_stream(stream, **kwags)
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
    oracle = from_stream(target)
    path = vmodel.make_piecewise_chord_progression(oracle, [chord_progression])
    return path

def path_to_stream(original, path, framesize=1.0, model_checker_state='s'):
    """Return a new stream from `original` following the path `offsets`.

    Keyword arguments:
        original_stream: music21.stream.Stream
            The stream on which to follow the path.
        path: int sequence or dict sequence
            The path given either:
                As a sequence of states within the oracle.
                As a sequence of dictionaries as output by model_checking,
                  in that case the key used to retrieve the sequence of states
                  in the oracle is `model_checker_state`.
        framesize: float, optional
            The duration of each frame in the sequence.
        model_checker_state: str
            The name of the variable in the model-checking used to represent
            the oracle's state.
    """
    new_stream = mus.stream.Stream()

    if not path:
        return new_stream
    elif isinstance(path[0], dict):
        path = [int(state[model_checker_state]) for state in path]
    
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

def progression_from_tonic(tonic, progression='authentic',
                           enable_motions=False, mode='major'):
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
    >>> progression_from_tonic('D', [8, '4', 5, '+1'], enable_motions=True)
    ['D', 'G', 'A', '+D']
    """
    p_tonic = mus.note.Note(tonic).pitchClass
    def make_degree(degree):
        """Auxiliary degree instantiation function.

        Return: (pitchClass, motion)
            pitchClass: int
                The pitch class relative to `degree` with given `p_tonic`.
            motion: None or str
                Eventual melodic motion if `degree` is prefixed by '+' or '-'. 
        """
        motion = None
        if isinstance(degree, str):
            if not enable_motions and degree[0] in ['+', '-']:
                raise ValueError("Must set enable_motions to True" +
                                 "to parse chord: {}".format(degree))
            if degree[0] == '+':
                motion = '+'
                degree = int(degree[1:])
            elif degree[0] == '-':
                motion = '-'
                degree = int(degree[1:])
            else:
                degree = int(degree)
        degree = ((degree - 1) % 7) + 1  # degree is now in range 1 .. 7

        result = None
        
        if degree == 1:
            result = p_tonic
        elif degree == 2:
            result = (p_tonic + 2) % 12
        elif degree == 3:
            if mode == 'major':
                result = (p_tonic + 4) % 12
            elif mode == 'minor':
                result = (p_tonic + 3) % 12
        elif degree == 4:
            result = (p_tonic + 5) % 12
        elif degree == 5:
            result = (p_tonic + 7) % 12
        elif degree == 6:
            if mode == 'major':
                result = (p_tonic + 9) % 12
            elif mode == 'minor':
                result = (p_tonic + 8) % 12
        elif degree == 7:
            result = (p_tonic + 11) % 12
        else:
            # degree is in the range 1 .. 7,
            # so coming here  means `mode` in unsupported
            raise ValueError("Mode {} is unsupported".format(mode))

        return result, motion

    if isinstance(progression, basestring):
        tonic_motion = ((p_tonic, '-') if enable_motions
                        else (p_tonic, None))
        if progression == 'authentic':
            classes = [make_degree(5), tonic_motion]
        elif progression == 'plagal':
            classes = [make_degree(4), tonic_motion]
        else:
            error_string = ("Cadence type {}".format(progression) +
                            "is unsupported (yet)")
            raise ValueError(error_string)
    elif isinstance(progression, list):
        classes = map(make_degree, progression)
    else:
        raise TypeError("Progression should be a string or a list of integers")

    pitch_names = [motion + mus.pitch.Pitch(p_class).name if motion
                   else mus.pitch.Pitch(p_class).name
                   for (p_class, motion) in classes]
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
    return extract_frame(stream.flat, offset_start, framesize)

def is_ascending_motion(chord1, chord2):
    """Return `True` if the motion from `chord1` to `chord2` is ascending.

    ----
    >>> cmaj4 = mus.chord.Chord(['C4', 'E4', 'G4', 'C5'])
    >>> cmaj5 = mus.chord.Chord(['C5', 'E4', 'G4'])
    >>> dmaj4 = mus.chord.Chord(['D4', 'F#4', 'A4', 'D5'])
    >>> is_ascending_motion(cmaj4, cmaj5) 
    True
    >>> is_ascending_motion(cmaj4, dmaj4)
    True
    >>> is_ascending_motion(dmaj4, cmaj4) 
    False
    """
    get_root = lambda chord: chord.sortAscending().root()
    if not chord1.pitches or not chord2.pitches:
        # If any of the chords is empty
        return None
    root1, root2 = map(get_root, (chord1, chord2))
    return root1 <= root2

if __name__ == "__main__":
    import doctest
    doctest.testmod()
