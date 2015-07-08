"""
bach_cello_midi_shuffle.py
example of symbolic oracle generation with VMO

Copyright (C) 11.05.2014 Cheng-i Wang

This file is part of Variable Markov Oracle (vmo) python library.

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
"""

import vmo
import vmo.generate as gen
import music21
import sys
import os

path_to_bach = (os.path.abspath('') + '/' +
                '../files/Suite_No_1_for_Cello_M1_Prelude.mxl')

def main(filepath=path_to_bach):
    """
    This example shows how to parse a music mxl file (music21 and
    musescore/finale required) and create a simple oracle representation.
    The output is a reshuffled midi stream shown in either musescore
    or finale based on your installation of music21. 
    
    OPTIONAL ARGS:
        seq_len: an integer for the length of the output sequence. 
        p: a float of the probability using the forward links.
        k: an integer for the starting state.
        LRS: an integer for the lower limit of the LRS of sfx/rsfx allowed
            to jump to.
        weight:
            None: choose uniformly among all the possible sfx/rsfx given 
                current state.
            "max": always choose the sfx/rsfx having the longest LRS.
            "weight": choose sfx/rsfx in a way that favors longer ones than 
            shorter ones.        
    """
    s = music21.converter.parse(filepath)
    # c = s.getElementById('Keyboard')
    m = s.flat.notes
    note_obj_seq = [x for x in m if type(x) is music21.note.Note]    
    oracle = vmo.build_oracle(note_obj_seq,'f')
    oracle.name = filepath.split('/')[-1]
    
    if len(sys.argv) == 1:
        b, kend, ktrace = gen.generate(oracle, len(note_obj_seq), 0.0, 0,
                                       LRS=2, weight='weight')
    else:
        seq_len = int(sys.argv[1])
        if seq_len == 0:
            seq_len = len(note_obj_seq)
        p = float(sys.argv[2])
        k = int(sys.argv[3])
        LRS = int(sys.argv[4])
        weight = sys.argv[5]
        b, kend, ktrace = gen.generate(oracle, seq_len, p, k,
                                       LRS=LRS, weight=weight)

    stream1 = music21.stream.Stream()
    x = [oracle.symbol[i] for i in b]
    for i in range(len(x)):
        _n = music21.note.Note(x[i].nameWithOctave)
        _n.duration.type = x[i].duration.type
        _n.duration = x[i].duration 
        stream1.append(_n)

    s.show()
    stream1.show()
    
if __name__ == '__main__':
    main()
