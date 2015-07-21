"""
tests/test_corpus.py
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

import music21 as mus

import vmo.VMO.oracle as voracle
import vmo.generate as vgen
import vmo.logics.model_checking as vmodel
import vmo.utils.chromagram as vchroma

import vmo.utils.nuxmv.model as nuxmv_model
import vmo.utils.nuxmv.check as nuxmv_check
import vmo.utils.nuxmv.properties as nuxmv_props

_BIG_INT = int(10000)

def oracle_from_corpus(name, framesize=1.0, threshold=0, suffix_method='inc',
                       weights=None, dfunc='tonnetz', dfunc_handle=None):
    # Unstable, can break depending on the structure of the piece extracted
    # from the corpus (e.g. if it's grouped in staves), use smart
    stream = mus.corpus.parse(name).flat.notes
    oracle = voracle.from_stream(stream, framesize=framesize,
                                 threshold=threshold,
                                 suffix_method=suffix_method,
                                 weights=weights,
                                 dfunc=dfunc, dfunc_handle=dfunc_handle)
    return oracle

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
    chord_progression = get_chord_progression(query)
    oracle = voracle.from_stream(target)
    path = vmodel.make_chord_progression(oracle, chord_progression)
    return path

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

def agglomerate_progression(progression):
    # TODO : this is not correct at all.
    def get_note(elem):
        if isinstance(elem, basestring):
            return elem
        else:
            name, _ = elem
            return name
    
    if not progression:
        return progression
    else:
        acc = (progression[0], 1)
        def aux(l, acc):
            if not l:
                note, start = acc
                infinity = _BIG_INT
                if start > 1:
                    l.append((note, (start, infinity)))
                    return l
                else:
                    l.append(note)
                    return l
            else:
                if l[0] == acc[0]:
                    acc = (acc[0], acc[1] + 1)
                    return aux(l[1:], acc)
                else:
                    new_acc = (l[0], 1)
                    note, start = acc
                    infinity = _BIG_INT
                    if start > 1:
                        result = aux(l[1:], new_acc)
                        result.append((note, (start, infinity)))
                        return result
                    else:
                        result = aux(l[1:], new_acc)
                        result.append(note)
                        return result
        result = aux(progression[1:], acc)
        result.reverse()
        return result
    
def make_chord_progression_tonic_free(oracle, progression, mode='major',
                                      start=None, include_rsfx=False,
                                      silence_equivalence=False,
                                      allow_init=False,
                                      model_checker='nuxmv'):
    """Return a path in `oracle` reaching the given `progression` from `start`.

    Tonic-free version: test for all 12 possibilities of instantiation of the
    degrees following the choice of an arbitrary tonic.
    The first existing path is returned.
    
    Keyword arguments:
        oracle: vmo.VMO.VMO
            The oracle on which to generate a path.
        progression: (int, int) sequence
            The chord progression to test for.
            Each pair in the sequence consists of:
                A pitch class.
                The length for which the note should be held, in quarter length.
        mode: string, optional
            The mode in which to generate the chord progression
            (default 'major').
        start: int, optional
            The index of the start from which the generated path should start
            (defaults to `oracle`'s initial state)
        include_rsfx: bool, optional
            Whether reverse suffix links should be included in the graph
            extracted from `oracle` (default `False`)
        silence_equivalence: bool, optional
            Whether silence should be considered equivalent to any given pitch
            (default False, since we then generates less uninteresting paths)
        allow_init: bool, optional
            Whether the initial state should be considered equivalent
            to any given pitch
            (default False, since we then generate less interesting paths)
    """
    if start is None:
        start = oracle.initial_state
    model, check, properties = nuxmv_model, nuxmv_check, nuxmv_props
    # vmodel._init_modules(model_checker)

    result = None
    
    tonic = 0
    model_str = model.print_oracle(oracle, init_state=start,
                                   include_rsfx=include_rsfx)
    while result is None and tonic < 12:
        tonic_name = mus.pitch.Pitch(tonic).name
        progression_inst = progression_from_tonic(
            tonic_name, progression=progression, mode=mode)
        progression_prop = properties.make_chord_progression(
            agglomerate_progression(progression_inst),
            exists=False,
            silence_equivalence=silence_equivalence)
        tonic += 1
        result = check.make_counterexample(model_str, progression_prop)
        
    return result


if __name__ == "__main__":
    import doctest
    import music21
    reload(nuxmv_props)
    doctest.testmod()
