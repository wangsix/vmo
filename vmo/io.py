"""io.py
offline factor/variable markov oracle generation routines for vmo

Copyright (C) 7.13.2015 Cheng-i Wang

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


def save_segments(outfile, boundaries, beat_intervals, labels=None):
    """Save detected segments to a .lab file.

    :parameters:
        - outfile : str
            Path to output file

        - boundaries : list of int
            Beat indices of detected segment boundaries

        - beat_intervals : np.ndarray [shape=(n, 2)]
            Intervals of beats

        - labels : None or list of str
            Labels of detected segments
    """

    if labels is None:
        labels = [('Seg#%03d' % idx) for idx in range(1, len(boundaries))]

    times = [beat_intervals[beat, 0] for beat in boundaries[:-1]]
    times.append(beat_intervals[-1, -1])

    with open(outfile, 'w') as f:
        for idx, (start, end, lab) in enumerate(zip(times[:-1],
                                                    times[1:],
                                                    labels), 1):
            f.write('%.3f\t%.3f\t%s\n' % (start, end, lab))


def save_motifs():
    pass


def save_oracle_meta():
    pass


