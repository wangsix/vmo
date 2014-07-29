'''
DrawOracle.py
drawing routines for PyOracle

Copyright (C) 12.02.2013 Greg Surges

This file is part of PyOracle.

PyOracle is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

PyOracle is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PyOracle.  If not, see <http://www.gnu.org/licenses/>.
'''

from random import randint

try:
    from PIL import Image, ImageDraw, ImageFilter #@UnresolvedImport @UnusedImport
except:
    print 'pil not loaded - hopefully running in max'

width = 900 * 4 
height = 400 * 4
oracle = None
image = None

lrs_threshold = 0

def start_draw(_oracle, size=(900*4, 400*4)):
    global oracle
    global image
    global width, height
    
    width = size[0]
    height = size[1]
    current_state = 0
    image = Image.new('RGB', (size[0], size[1]))
    oracle = _oracle 
    return draw(current_state)

def draw(current_state):
    
    if type(oracle) == dict:
        trn = oracle['trn']
        sfx = oracle['sfx']
        lrs = oracle['lrs']
        rsfx = oracle['rsfx']
    else:
        trn = oracle.trn
        sfx = oracle.sfx
        lrs = oracle.lrs
        rsfx = oracle.rsfx
    
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
                draw.line((current_x, height/2, next_x, height/2), width=1,fill='white')
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
