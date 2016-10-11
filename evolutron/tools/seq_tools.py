# coding=utf-8

import numpy as np

"""
Dictionaries
"""
aa_map = {
    'M': 0,
    'L': 1,
    'G': 2,
    'P': 3,
    'A': 4,
    'V': 5,
    'I': 6,
    'C': 7,
    'F': 8,
    'Y': 9,
    'W': 10,
    'H': 11,
    'K': 12,
    'R': 13,
    'Q': 14,
    'N': 15,
    'E': 16,
    'D': 17,
    'S': 18,
    'T': 19,
}

aa_map_rev = {
    0: 'M',
    1: 'L',
    2: 'G',
    3: 'P',
    4: 'A',
    5: 'V',
    6: 'I',
    7: 'C',
    8: 'F',
    9: 'Y',
    10: 'W',
    11: 'H',
    12: 'K',
    13: 'R',
    14: 'Q',
    15: 'N',
    16: 'E',
    17: 'D',
    18: 'S',
    19: 'T'
}

nt_map = {
    #       A    C    G    T
    'A': [1.0, 0.0, 0.0, 0.0],
    'C': [0.0, 1.0, 0.0, 0.0],
    'G': [0.0, 0.0, 1.0, 0.0],
    'T': [0.0, 0.0, 0.0, 1.0],
    'Y': [0.0, 0.5, 0.0, 0.5],
    'R': [0.5, 0.0, 0.5, 0.0],
    'K': [0.0, 0.0, 0.5, 0.5],
    'M': [0.5, 0.5, 0.0, 0.0],
    'S': [0.0, 0.5, 0.5, 0.0],
    'W': [0.5, 0.0, 0.0, 0.5],
    'B': [0.0, 0.33, 0.33, 0.33],
    'D': [0.33, 0.0, 0.33, 0.33],
    'H': [0.33, 0.33, 0.0, 0.33],
    'V': [0.33, 0.33, 0.33, 0.0],
    'N': [0.25, 0.25, 0.25, 0.25],
}

"""
Transformations between representations
"""


def aa2num(aa_seq):
    """ Transforms an amino acid sequence into a float representation. """
    return [aa_map[aa] for aa in
            aa_seq.replace('X', 'M').replace('Z', 'Q').replace('B', 'N').replace('U', 'S').replace('O', 'K')]


def num2aa(num_seq):
    return ''.join([aa_map_rev[n] for n in num_seq])


def num2hot(num):
    return np.eye(20, dtype=np.float32)[num].T


def hot2num(hot):
    return [np.argmax(h) for h in hot.T]


def aa2hot(aa_seq):
    return num2hot(aa2num(aa_seq))


def nt2prob(seq):
    nt = []

    for s in seq:
        nt.append(nt_map[s])

    return np.asarray(nt, dtype=np.float32)


def ntround(x):
    assert (x >= 0)
    assert (x <= 1)
    x = round(x, 2)
    if x <= 0.2:
        return 0.0
    elif x <= 0.4:
        return 0.33
    elif x <= 0.6:
        return 0.5
    else:
        return 1.0


def prob2nt(prob):
    assert (isinstance(prob, np.ndarray))

    prob = prob.reshape((-1, 4))

    f = np.vectorize(ntround)

    return f(prob)
