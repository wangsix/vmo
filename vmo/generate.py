"""generate.py
offline audio oracle / factor oracle generation routines for vmo

Copyright (C) 12.02.2013 Greg Surges, Cheng-i Wang

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

import random
import itertools
import numpy as np
import librosa


def improvise_step(oracle, i, lrs=0, weight=None, prune=False):
    """ Given the current time step, improvise (generate) the next time step based on the oracle structure.

    :param oracle: an indexed vmo object
    :param i: current improvisation time step
    :param lrs: the length of minimum longest repeated suffixes allowed to jump
    :param weight: if None, jump to possible candidate time step uniformly, if "lrs", the probability is proportional
    to the LRS of each candidate time step
    :param prune: whether to prune improvisation steps based on regular beat structure or not
    :return: the next time step
    """

    latent = oracle.latent[int(oracle.data[i])]
    if prune:
        prune_list = range(i % prune, oracle.n_states - 1, prune)
        trn_link = [s + 1 for s in latent if
                    (oracle.lrs[s] >= lrs and
                     (s + 1) < oracle.n_states) and
                    s in prune_list]
    else:
        trn_link = [s + 1 for s in latent if
                    (oracle.lrs[s] >= lrs and (s + 1) < oracle.n_states)]
    if not trn_link:
        if i == oracle.n_states - 1:
            n = 1
        else:
            n = i + 1
    else:
        if weight == 'lrs':
            lrs_link = [oracle.lrs[s] for s in latent if
                        (oracle.lrs[s] >= lrs and (s + 1) < oracle.n_states)]
            lrs_pop = list(itertools.chain.from_iterable(itertools.chain.from_iterable(
                [[[i] * _x for (i, _x) in zip(trn_link, lrs_link)]])))
            n = np.random.choice(lrs_pop)
        else:
            n = trn_link[int(np.floor(random.random() * len(trn_link)))]
    return n


def improvise(oracle, seq_len, k=1, LRS=0, weight=None, continuity=1):
    """ Given an oracle and length, generate an improvised sequence of the given length.

    :param oracle: an indexed vmo object
    :param seq_len: the length of the returned improvisation sequence
    :param k: the starting improvisation time step in oracle
    :param LRS: the length of minimum longest repeated suffixes allowed to jump
    :param weight: if None, jump to possible candidate time step uniformly, if "lrs", the probability is proportional
    to the LRS of each candidate time step
    :param continuity: the number of time steps guaranteed to continue before next jump is executed
    :return: the improvised sequence
    """

    s = []
    if k + continuity < oracle.n_states - 1:
        s.extend(range(k, k + continuity))
        k = s[-1]
        seq_len -= continuity

    while seq_len > 0:
        s.append(improvise_step(oracle, k, LRS, weight))
        k = s[-1]
        if k + 1 < oracle.n_states - 1:
            k += 1
        else:
            k = 1
        if k + continuity < oracle.n_states - 1:
            s.extend(range(k, k + continuity))
            seq_len -= continuity
        k = s[-1]
        seq_len -= 1

    return s


def markov_improvise():
    pass


def generate(oracle, seq_len, p=0.5, k=1, LRS=0, weight=None):
    """ Generate a sequence based on traversing an oracle.

    :param oracle: a indexed vmo object
    :param seq_len: the length of the returned improvisation sequence
    :param p: a float between (0,1) representing the probability using the forward links.
    :param k: the starting improvisation time step in oracle
    :param LRS: the length of minimum longest repeated suffixes allowed to jump
    :param weight:
            None: choose uniformly among all the possible sfx/rsfx given
                current state.
            "max": always choose the sfx/rsfx having the longest LRS.
            "weight": choose sfx/rsfx in a way that favors longer ones than
            shorter ones.
    :return:
            s: a list containing the sequence generated, each element represents a
            state.
            kend: the ending state.
            ktrace:
    """

    trn = oracle.trn[:]
    sfx = oracle.sfx[:]
    lrs = oracle.lrs[:]
    rsfx = oracle.rsfx[:]

    s = []
    ktrace = [1]

    for _i in range(seq_len):
        # generate each state
        if sfx[k] != 0 and sfx[k] is not None:
            if (random.random() < p):
                # copy forward according to transitions
                I = trn[k]
                if len(I) == 0:
                    # if last state, choose a suffix
                    k = sfx[k]
                    ktrace.append(k)
                    I = trn[k]
                sym = I[int(np.floor(random.random() * len(I)))]
                s.append(sym)  # Why (sym-1) before?
                k = sym
                ktrace.append(k)
            else:
                # copy any of the next symbols
                ktrace.append(k)
                _k = k
                k_vec = []
                k_vec = _find_links(k_vec, sfx, rsfx, _k)
                k_vec = [_i for _i in k_vec if lrs[_i] >= LRS]
                lrs_vec = [lrs[_i] for _i in k_vec]
                if len(k_vec) > 0:  # if a possibility found, len(I)
                    if weight == 'weight':
                        max_lrs = np.amax(lrs_vec)
                        query_lrs = max_lrs - np.floor(random.expovariate(1))

                        if query_lrs in lrs_vec:
                            _tmp = np.where(lrs_vec == query_lrs)[0]
                            _tmp = _tmp[int(
                                np.floor(random.random() * len(_tmp)))]
                            sym = k_vec[_tmp]
                        else:
                            _tmp = np.argmin(abs(
                                np.subtract(lrs_vec, query_lrs)))
                            sym = k_vec[_tmp]
                    elif weight == 'max':
                        sym = k_vec[np.argmax([lrs[_i] for _i in k_vec])]
                    else:
                        sym = k_vec[int(np.floor(random.random() * len(k_vec)))]

                    if sym == len(sfx) - 1:
                        sym = sfx[sym] + 1
                    else:
                        s.append(sym + 1)
                    k = sym + 1
                    ktrace.append(k)
                else:  # otherwise continue
                    if k < len(sfx) - 1:
                        sym = k + 1
                    else:
                        sym = sfx[k] + 1
                    s.append(sym)
                    k = sym
                    ktrace.append(k)
        else:
            if k < len(sfx) - 1:
                s.append(k + 1)
                k += 1
                ktrace.append(k)
            else:
                sym = sfx[k] + 1
                s.append(sym)
                k = sym
                ktrace.append(k)
        if k >= len(sfx) - 1:
            k = 0
    kend = k
    return s, kend, ktrace


def _find_links(k_vec, sfx, rsfx, k):
    """Find sfx/rsfx recursively."""
    k_vec.sort()
    if 0 in k_vec:
        return k_vec
    else:
        if sfx[k] not in k_vec:
            k_vec.append(sfx[k])
        for i in range(len(rsfx[k])):
            if rsfx[k][i] not in k_vec:
                k_vec.append(rsfx[k][i])
        for i in range(len(k_vec)):
            k_vec = _find_links(k_vec, sfx, rsfx, k_vec[i])
            if 0 in k_vec:
                break
        return k_vec


def _make_win(n, mono=False):
    """ Generate a window for a given length.

    :param n: an integer for the length of the window.
    :param mono: True for a mono window, False for a stereo window.
    :return: an numpy array containing the window value.
    """

    if mono:
        win = np.hanning(n) + 0.00001
    else:
        win = np.array([np.hanning(n) + 0.00001, np.hanning(n) + 0.00001])
    win = np.transpose(win)
    return win


def audio_synthesis(ifilename, ofilename, s, analysis_sr=44100, buffer_size=8192, hop=4096):
    """

    :param ifilename: input audio file path.
    :param ofilename: output audio file path.
    :param s: frame sequence to be generated.
    :param analysis_sr: the sampling frequency of the ifilename.
    :param buffer_size: should match fft/frame size of oracle analysis.
    :param hop: hop size, should be 1/2 the buffer_size.
    :return: the improvised sequence in audio wave file
    """
    x, fs = librosa.load(ifilename, sr=analysis_sr)

    if fs != analysis_sr:
        buffer_size *= (fs / float(analysis_sr))
        buffer_size = int(buffer_size)
        hop *= (fs / float(analysis_sr))
        hop = int(hop)

    mono = True
    if x.ndim == 2:
        mono = False
    xmat = []
    for i in range(0, len(x), hop):
        if i + buffer_size >= len(x):
            if mono:
                x = np.append(x, np.zeros((i + buffer_size - len(x),)))
            else:
                x = np.vstack((x, np.zeros((i + buffer_size - len(x), 2))))
        new_mat = np.array(x[i:i + buffer_size])  # try changing array type?
        xmat.append(new_mat)
    xmat = np.array(xmat)

    s = np.array(s) - 1
    xnewmat = xmat[s]

    framelen = len(xnewmat[0])
    nframes = len(xnewmat)
    win = _make_win(framelen, mono)

    if mono:
        wsum = np.zeros(((nframes - 1) * hop + framelen,))
        x_new = np.zeros(((nframes - 1) * hop + framelen,))
    else:
        wsum = np.zeros(((nframes - 1) * hop + framelen, 2))
        x_new = np.zeros(((nframes - 1) * hop + framelen, 2))

    win_pos = range(0, len(x_new), hop)
    for i in range(0, nframes):
        len_xnewmat = len(xnewmat[i])
        x_new[win_pos[i]:win_pos[i] + len_xnewmat] = np.add(
            x_new[win_pos[i]:win_pos[i] + len_xnewmat],
            np.multiply(xnewmat[i], win))
        wsum[win_pos[i]:win_pos[i] + len_xnewmat] = np.add(
            wsum[win_pos[i]:win_pos[i] + len_xnewmat],
            win)
    x_new[hop:-hop] = np.divide(x_new[hop:-hop], wsum[hop:-hop])
    x_new = x_new.astype(np.float32)
    librosa.output.write_wav(path=ofilename, y=x_new, sr=analysis_sr)
    return x_new, wsum, fs


def generate_audio(ifilename, ofilename, oracle, seq_len,
                   analysis_sr=44100, buffer_size=8192, hop=4096,
                   p=0.5, k=0, lrs=0):
    """

    :param ifilename: input audio file path.
    :param ofilename: output audio file path.
    :param oracle: an oracle indexed on ifilename
    :param seq_len: length of sequence to be generated, in frames.
    :param analysis_sr: the sampling frequency of the ifilename.
    :param buffer_size: should match fft/frame size of oracle analysis.
    :param hop: hop size, should be 1/2 the buffer_size.
    :param p: continuity parameter.
    :param k: start frame number.
    :param lrs: the length of minimum longest repeated suffixes allowed to jump
    :return: the improvised sequence in audio wave file
    """

    x, fs = librosa.load(ifilename, sr=analysis_sr)

    if fs != analysis_sr:
        buffer_size *= (fs / float(analysis_sr))
        buffer_size = int(buffer_size)
        hop *= (fs / float(analysis_sr))
        hop = int(hop)

    mono = True
    if x.ndim == 2:
        mono = False
    xmat = []
    for i in range(0, len(x), hop):
        if i + buffer_size >= len(x):
            if mono:
                x = np.append(x, np.zeros((i + buffer_size - len(x),)))
            else:
                x = np.vstack((x, np.zeros((i + buffer_size - len(x), 2))))
        new_mat = np.array(x[i:i + buffer_size])  # try changing array type?
        xmat.append(new_mat)
    xmat = np.array(xmat)

    s, _kend, _ktrace = generate(oracle, seq_len, p, k, lrs)
    s = np.array(s) - 1
    xnewmat = xmat[s]

    framelen = len(xnewmat[0])
    nframes = len(xnewmat)
    win = _make_win(framelen, mono)

    if mono:
        wsum = np.zeros(((nframes - 1) * hop + framelen,))
        x_new = np.zeros(((nframes - 1) * hop + framelen,))
    else:
        wsum = np.zeros(((nframes - 1) * hop + framelen, 2))
        x_new = np.zeros(((nframes - 1) * hop + framelen, 2))

    win_pos = range(0, len(x_new), hop)
    for i in range(0, nframes):
        len_xnewmat = len(xnewmat[i])
        x_new[win_pos[i]:win_pos[i] + len_xnewmat] = np.add(
            x_new[win_pos[i]:win_pos[i] + len_xnewmat],
            np.multiply(xnewmat[i], win))
        wsum[win_pos[i]:win_pos[i] + len_xnewmat] = np.add(
            wsum[win_pos[i]:win_pos[i] + len_xnewmat],
            win)
    x_new[hop:-hop] = np.divide(x_new[hop:-hop], wsum[hop:-hop])
    # x_new = x_new.astype(np.int16)
    librosa.output.write_wav(path=ofilename, y=x_new, sr=analysis_sr)
    return x_new, wsum, fs
