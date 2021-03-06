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
    'U': 20,
    'O': 21,
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
    19: 'T',
    20: 'U',
    21: 'O',
    22: '-'
}

smiles_map = {
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
    'U': 20,
    'O': 21,
}

SecS_map_8cat = {
    ' ': 0,
    '-': 0,
    'L': 0,
    'C': 0,
    'H': 1,
    'G': 2,
    'T': 3,
    'S': 4,
    'B': 5,
    'E': 6,
    'I': 7,
}

SecS_map_8cat_rev = {
    0: 'C',
    1: 'H',
    2: 'G',
    3: 'T',
    4: 'S',
    5: 'B',
    6: 'E',
    7: 'I',
}

SecS_map_3cat = {
    ' ': 0,
    'L': 0,
    'C': 0,
    'H': 1,
    'G': 1,
    'T': 0,
    'S': 0,
    'B': 2,
    'E': 2,
    'I': 1,
}

SecS_map_3cat_rev = {
    0: 'C',
    1: 'H',
    2: 'E',
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

aa2cod_map = {
    'M': [11],
    'L': range(2, 8),
    'G': range(60, 64),
    'P': range(20, 24),
    'A': range(28, 32),
    'V': range(12, 16),
    'I': range(8, 11),
    'C': range(48, 50),
    'F': [0, 1],
    'Y': range(32, 34),
    'W': [51],
    'H': range(36, 38),
    'K': range(42, 44),
    'R': [52, 53, 54, 55, 58, 59],
    'Q': range(38, 40),
    'N': range(40, 42),
    'E': range(46, 48),
    'D': range(44, 46),
    'S': [16, 17, 18, 19, 56, 57],
    'T': range(24, 28),
    'U': [50],
    'O': [50],
}

"""
Transformations between representations
"""


def aa2hot(aa_seq, n=20):
    if n == 20:
        aa_map.pop('U', 0)
        aa_map.pop('O', 0)
    hot = np.zeros(shape=(len(aa_seq), n))
    for idx, aa in enumerate(aa_seq):
        if aa in aa_map:
            hot[idx, aa_map[aa]] = 1
        elif aa == 'X':
            hot[idx, :] = [1 / n for _ in range(n)]
        elif aa == 'B':
            hot[idx, aa_map['D']] = .5
            hot[idx, aa_map['N']] = .5
        elif aa == 'Z':
            hot[idx, aa_map['E']] = .5
            hot[idx, aa_map['Q']] = .5
        elif aa == 'J':
            hot[idx, aa_map['I']] = .5
            hot[idx, aa_map['L']] = .5
    return hot


def hot2aa(hot):
    num = [np.argmax(h) for h in hot if np.sum(h) != 0.0]
    pad = [22 for h in hot if np.sum(h) == 0.0]
    return ''.join([aa_map_rev[n] for n in num+pad])


def secs2hot(sec_seq, cats=3):
    if cats == 3:
        num = [SecS_map_3cat[s] for s in sec_seq]
    elif cats == 8:
        num = [SecS_map_8cat[s] for s in sec_seq]
    else:
        raise ValueError('Invalid option for secondary structure . Should be 8 or 3.')
    return np.eye(cats, dtype=np.float32)[num]


# TODO: refactor this
def hot2SecS_8cat(hot):
    num = [np.argmax(h) for h in hot]

    return ''.join([SecS_map_8cat_rev[n] for n in num])


def hot2SecS_3cat(hot):
    num = [np.argmax(h) for h in hot]

    return ''.join([SecS_map_3cat_rev[n] for n in num])


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


def aa2codon(aa_seq):
    cod = np.zeros((len(aa_seq), 64))
    for i in range(len(aa_seq)):
        cod[i, aa2cod_map[aa_seq[i]]] = 1

    return cod
