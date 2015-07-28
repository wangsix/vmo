"""
utils/chromagram.py
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
from math import floor, ceil
from scipy.ndimage.filters import gaussian_filter1d as gaussian

import music21 as mus

"""Music21 chords and streams to chromagram conversion.

This module exports a function to turn a music21 Chord/Note object
into a chromagram (a 12 dimensional array of pitch classes) and
extends to general Stream objects (returning a matrix of chromagrams).
"""    

pitch_space_size = 12

def _from_pitch(pitch):
    """Return chromagram for a single, low-level Pitch object.

    Keyword aguments:
        pitch: music21.pitch.Pitch
            The pitch to convert
    ----
    >>> _from_pitch(music21.pitch.Pitch('D'))
    array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    >>> _from_pitch(music21.pitch.Pitch('E3'))
    array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    """
    p_class = pitch.pitchClass
    chroma = np.zeros(pitch_space_size,dtype=np.int64)
    chroma[p_class] = 1
    return chroma

def _from_note(note):
    """Return chromagram for a single Note object.

    Keyword aguments:
        note: music21.note.Note
            The note to convert
    ----
    >>> _from_note(music21.note.Note('E4'))
    array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    """
    return _from_pitch(note.pitch)

def from_chord(chord):
    """Return chromagram for a Chord object.

    Discard multiple occurences of a given note wrt enharmonic
    and octave equivalence, keeping only one.

    Keyword aguments:
        chord: music21.chord.Chord
            The chord to convert
    ----
    >>> chord = music21.chord.Chord(['D3', 'F#3', 'A3', 'D4'])
    >>> from_chord(chord)
    array([0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0])
    """
    if isinstance(chord, mus.note.Note):
        return _from_note(chord)
    elif isinstance(chord, mus.chord.Chord):
        p_classes = chord.orderedPitchClasses
        chroma = np.zeros(pitch_space_size,dtype=np.int64)
        for p_class in p_classes:
            chroma[p_class] = 1
        return chroma
    else:
        raise ValueError("Not a chord of note")

def to_chord(chroma):
    """Return music21 Chord for a chromagram array.

    Return only pitches, no octaves.

    Keyword aguments:
        chroma: np.array
            The chromagram to convert
    ----
    >>> chroma = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0])
    >>> result = to_chord(chroma)
    >>> expected = music21.chord.Chord(['D', 'F#', 'A'])

    Test normalForm equality, because Chord object have unique identifiers
    >>> result.normalForm == expected.normalForm
    True
    """
    pitches = []
    def make_note(pitchClass):
        return mus.note.Note(mus.pitch.Pitch(pitchClass))
    for i in range(12):
        if chroma[i] > 0:
            pitches.append(make_note(i))
    return mus.chord.Chord(pitches)
            
def from_stream(stream, framesize=1.0, overlap=0.0,
                smooth=False, sigma=None):
    """Slice stream at all quarter lengths and return the matrix of chromagrams

    Keyword arguments:
        stream: music21.stream.Stream
            the input stream
        framesize: float
            the quarter-length of each frames in the sliced stream
            (default 1.0)
        overlap: float, optional
            the overlap to introduce, in quarter length (default 0.0)
        smooth: bool
            Whether the output should be smoothed.
            If True, applies a row-to-row gaussian-filter with the given
            value of sigma with the effect of smoothing the content over time
        sigma: float, optional
            the value to use for the gaussian filter
            (default None, value is then set according to framesize)

    TODO: Check behaviour with variable tempo
    ----
    >>> n1 = mus.note.Note('C', quarterLength=4)
    >>> n2 = mus.chord.Chord(['E', 'G', 'B'], quarterLength=4)
    >>> s = mus.stream.Stream([n1])
    >>> s.append(n2)
    >>> chromagram = from_stream(s)
    
    s holds a 'C' on the 4th beat of the first measure:  
    >>> chromagram[0, 3] == 1
    True

    s holds an 'E' and a 'G' on the first beat of the second measure
        (offset of 4 quarters from the beginning):
    >>> chromagram[4, 4] == chromagram[7, 4] == 1
    True

    s holds no 'C' on the first beat of the second measure:
    >>> chromagram[0, 4] == 0
    True
    """
    import vmo.utils.music21_interface as vmusic

    chords = stream.flat.chordify().notes
    duration_quarters = int(floor(chords.duration.quarterLength))
    frames_count = int(ceil(duration_quarters * (1. / framesize)))
    offsets = np.arange(0, duration_quarters, framesize)
    
    # slice the chordified stream uniformly at each framesize
    chords = chords.sliceAtOffsets(offsets)

    chroma_matrix = np.zeros(shape=(pitch_space_size, frames_count))
    for frame, offset in enumerate(offsets):
        elements = vmusic.extract_frame(chords, offset, framesize)
        # Get all pitch classes appearing in the extracted frame
        pitch_classes = [pc for chord in elements for pc in chord.pitchClasses]
                                       
        chroma_vector = from_chord(mus.chord.Chord(pitch_classes))
            
        chroma_matrix[:,frame] = chroma_vector
        
    if smooth:
        if sigma is None:
            sigma = 4*framesize  # Arbitrary value. In the scale of `framesize`
        for i in range(pitch_space_size):
            smoothed_row = gaussian(chroma_matrix[i,:], sigma=sigma)
            chroma_matrix[i,:] = smoothed_row

    return chroma_matrix


if __name__ == "__main__":
    import doctest
    import music21
    doctest.testmod()
