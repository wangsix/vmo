"""
plot.py
drawing routines for vmo

Copyright (C) 8.20.2014 Cheng-i Wang

This file is part of vmo.

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

import numpy as np
import music21
import pretty_midi

try:
    from PIL import Image, ImageDraw, ImageFilter #@UnresolvedImport @UnusedImport
except:
    print('pil not loaded - hopefully running in max')

width = 900 * 4 
height = 400 * 4
lrs_threshold = 0


def start_draw(_oracle, size=(900*4, 400*4)):
    """

    :param _oracle: input vmo object
    :param size: the size of the output image
    :return: an update call the draw()
    """
    
    width = size[0]
    height = size[1]
    current_state = 0
    image = Image.new('RGB', (width, height))
    oracle = _oracle 
    return draw(oracle, current_state, image, width, height)


def draw(oracle, current_state, image, width=width, height=height):
    """

    :param oracle: input vmo object
    :param current_state:
    :param image: an PIL image object
    :param width: width of the image
    :param height: height of the image
    :return: the updated PIL image object
    """
    
    trn = oracle.trn
    sfx = oracle.sfx
    lrs = oracle.lrs
    
    # handle to Draw object - PIL
    N_states = len(sfx)
    draw = ImageDraw.Draw(image)
        
    for i in range(N_states):
        # draw circle for each state
        x_pos = (float(i) / N_states * width) + 0.5 * 1.0 / N_states * width
        # iterate over forward transitions
        for tran in trn[i]:
            # if forward transition to next state
            if tran == i + 1:
                # draw forward transitions
                next_x = (float(i + 1) / N_states * width) + 0.5 * 1.0 / N_states * width
                current_x = x_pos + (0.25 / N_states * width)
                draw.line((current_x, height/2, next_x, height/2), width=1, fill='white')
            else:
                if lrs[tran] >= lrs_threshold:
                    # forward transition to another state
                    current_x = x_pos
                    next_x = (float(tran) / N_states * width) + (0.5 / N_states * width)
                    arc_height = (height / 2) + (tran - i) * 0.125
                    draw.arc((int(current_x), int(height/2 - arc_height/2),
                        int(next_x), int(height/2 + arc_height / 2)), 180, 0,
                        fill='White')
        if sfx[i] is not None and sfx[i] != 0 and lrs[sfx[i]] >= lrs_threshold:
            current_x = x_pos
            next_x = (float(sfx[i]) / N_states * width) + (0.5 / N_states * width)
            # draw arc
            arc_height = (height / 2) - (sfx[i] - i) * 0.125
            draw.arc((int(next_x), 
                      int(height/2 - arc_height/2), 
                      int(current_x), 
                      int(height/2 + arc_height/2)),
                      0,
                      180,
                      fill='White')
        
    image.resize((900, 400), (Image.BILINEAR))
    return image


def draw_compror():
    raise NotImplementedError("Compror drawing is under construction, coming soon!")


def get_pattern_mat(oracle, pattern):
    """Output a matrix containing patterns in rows from a vmo.

    :param oracle: input vmo object
    :param pattern: pattern extracted from oracle
    :return: a numpy matrix that could be used to visualize the pattern extracted.
    """

    pattern_mat = np.zeros((len(pattern), oracle.n_states-1))
    for i,p in enumerate(pattern):
        length = p[1]
        for s in p[0]:
            pattern_mat[i][s-length:s-1] = 1
    
    return pattern_mat


def plot_midi_frame(midi_data, beat_positions, frame_ind):

    beat_start = beat_positions[frame_ind]
    beat_end = beat_positions[frame_ind + 1]
    n_list = []
    for i in midi_data.instruments:
        if not i.is_drum:
            for n in i.notes:
                if (n.start >= beat_start) & (n.start < beat_end) \
                        or (n.end >= beat_start) & (n.end < beat_end)\
                        or (n.start <= beat_start) & (n.end > beat_end):
                    note = music21.note.Note(pretty_midi.utilities.note_number_to_name(n.pitch))
                    if not note in n_list:
                        n_list.append(note)

    chord = music21.chord.Chord(n_list)
    return chord


def plot_chroma_as_chord(chroma_frame, n_pitch=3):

    pitch_rank = np.argsort(chroma_frame)
    n_list = []
    for p in pitch_rank[-n_pitch:]:
        note = pretty_midi.utilities.note_number_to_name(p + 60)
        n_list.append(note)
    chroma = music21.chord.Chord(n_list)

    return chroma



