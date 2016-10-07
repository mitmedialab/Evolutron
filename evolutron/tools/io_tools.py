# coding=utf-8
try:
    import cPickle
except:
    import pickle as cPickle
import csv

import numpy as np
from Bio import SeqIO
import gzip

from evolutron.tools.seq_tools import aa_map, nt_map, num2hot


############################
# -------- Classes ------- #
############################
class Protein(object):
    def __init__(self, name, organism, aa_seq, nt_seq):
        self.name = name
        self.aa_seq = aa_seq
        self.nt_seq = nt_seq
        self.aa = str(self.aa_seq)
        self.nt = str(self.nt_seq)
        self.aa_num = len(aa_seq)
        if nt_seq:
            self.nt_num = len(nt_seq)
        else:
            self.nt_num = None
        self.org = organism


class ZincFinger(Protein):
    def __init__(self, name, organism, aa_seq, nt_seq, pwm):
        """

        :type rec_site: Nucleotide sequence of the recognition site
        """
        Protein.__init__(self, name, organism, aa_seq, nt_seq)
        self.pwm = pwm
        self.rec_site = np.power(np.ones_like(pwm) * 2.0, pwm) * .25

    def __str__(self):
        return '{x.name} {x.nt_num} nt {x.aa_num} aa'.format(x=self)


############################
# -------- Functions ------- #
############################


def b1h(padded=False, min_aa=None, max_aa=None):
    filename = 'datasets/B1H.motifs.csv'

    data = csv.DictReader(open(filename, 'r'))

    znf = []

    for d in data:
        r_s = np.zeros((4, 4))
        for i, nt in enumerate(['A', 'C', 'G', 'T']):
            for pos in range(4):
                r_s[i, pos] = d[nt + str(pos + 1)]
        znf.append(ZincFinger(d['Gene/Range'], d['Organism'], d['ZF protein sequence'], None, r_s))

    aa_list = [zf.aa for zf in znf if zf.aa.find('X') == -1]

    if padded:
        if not max_aa:
            max_aa = max(map(len, aa_list))
        x_data = np.zeros((len(aa_list), 20, max_aa), dtype=np.float32)
        for i, aa_seq in enumerate(aa_list):
            for j, aa in enumerate(aa_seq):
                x_data[i, int(aa_map[aa]), j] = 1
    else:
        x_data = [np.zeros((20, max(len(aa_seq), min_aa)), dtype=np.float32) for aa_seq in aa_list]

        for ind, aa_seq in enumerate(aa_list):
            for j, aa in enumerate(aa_seq):
                x_data[ind][int(aa_map[aa]), j] = 1

    y_data = np.asarray([zf.rec_site.reshape(16) for zf in znf if zf.aa.find('X') == -1])

    data_set = x_data, y_data

    return data_set


def m6a(padded=False, min_aa=None, max_aa=None, probe='both'):
    """
        This module parses data from m6a proto-array for Evolutron
    """
    infile = 'datasets/m6a.csv'
    data = csv.DictReader(open(infile, 'r'))
    data = list(data)

    empty = [d for d in data if len(d['AA']) <= 4]

    unknown = [d for d in data if not d['AA'].find('U') == -1 or not d['AA'].find('X') == -1]

    passed = [d for d in data if len(d['AA']) > 5 and d['AA'].find('U') == -1 and d['AA'].find('X') == -1]

    probe1 = [p for p in passed[::2]]
    probe2 = [p for p in passed[1::2]]

    if probe == '1':
        datapoints = probe1
    elif probe == '2':
        datapoints = probe2
    else:
        datapoints = passed

    del probe1
    del probe2
    del passed

    aa_list = []
    conc_list = []
    for d in datapoints:
        aa_list.append(eval(d['AA'])[0][0])
        conc_list.append([float(d['Array_60531']), float(d['Array_60690']), float(d['mean'])])

    if padded:
        if not max_aa:
            max_aa = max(map(len, aa_list))
        x_data = np.zeros((len(aa_list), 20, max_aa), dtype=np.float32)
        for i, aa_seq in enumerate(aa_list):
            for j, aa in enumerate(aa_seq):
                if j == min_aa:
                    break
                x_data[i, int(aa_map[aa]), j] = 1
    else:
        x_data = [np.zeros((20, max(len(aa_seq), min_aa)), dtype=np.float32) for aa_seq in aa_list]

        for ind, aa_seq in enumerate(aa_list):
            for j, aa in enumerate(aa_seq):
                x_data[ind][int(aa_map[aa]), j] = 1

    y_data = np.asarray(conc_list)

    data_set = x_data, y_data

    return data_set


def type2p(padded=False, min_aa=None, max_aa=None):
    """
        This module parses data from REBASE and transforms them for Evolutron
    """
    # Read raw input
    f = open('datasets/type2p_and_ps.in', 'r')
    input_raw = cPickle.load(f)

    # Transform raw input into data-set
    aa_list = []
    nt_list = []
    for enz in input_raw:
        if enz.seq.find('X') == -1:
            aa_list.append(str(enz.seq))
            nt_list.append(enz.description)

    if padded:
        if not max_aa:
            max_aa = max(map(len, aa_list))
        x_data = np.zeros((len(aa_list), 20, max(min_aa, max_aa)), dtype=np.float32)
        for i, aa_seq in enumerate(aa_list):
            for j, aa in enumerate(aa_seq):
                if j > max_aa - 1:
                    break
                x_data[i, int(aa_map[aa]), j] = 1
    else:
        if not max_aa:
            max_aa = max(map(len, aa_list))
        x_data = [np.zeros((20, min(max(len(aa_seq), min_aa), max_aa)), dtype=np.float32) for aa_seq in aa_list]

        for ind, aa_seq in enumerate(aa_list):
            for j, aa in enumerate(aa_seq):
                if j > max_aa - 1:
                    break
                x_data[ind][int(aa_map[aa]), j] = 1

    y_data = [np.zeros((len(nt_seq), 4), dtype=np.float32) for nt_seq in nt_list]
    for ind, nt_seq in enumerate(nt_list):
        for j, nt in enumerate(nt_seq):
            y_data[ind][j, :] = np.asarray(nt_map[nt])
        y_data[ind] = y_data[ind].reshape(24)

    data_set = x_data, y_data

    return data_set


def fasta_parser(filename, padded=False, min_aa=None, max_aa=None):
    """
        This module parses data from FASTA files and transforms them to Evolutron format.
    """

    input_file = open(filename, "rU")

    aa_list = []
    for record in SeqIO.parse(input_file, "fasta"):

        seq = str(record.seq)

        seq = seq.replace('X', 'M').replace('Z', 'Q').replace('B', 'N').replace('U', 'S').replace('O', 'K')

        if len(seq) < 1500:
            aa_list.append(seq)

    if padded:
        if not max_aa:
            max_aa = max(map(len, aa_list))
        x_data = np.zeros((len(aa_list), 20, max(min_aa, max_aa)), dtype=np.float32)
        for i, aa_seq in enumerate(aa_list):
            for j, aa in enumerate(aa_seq):
                if j > max_aa - 1:
                    break
                x_data[i, int(aa_map[aa]), j] = 1
    else:
        if not max_aa:
            max_aa = max(map(len, aa_list))
        x_data = [np.zeros((20, min(max(len(aa_seq), min_aa), max_aa)), dtype=np.float32) for aa_seq in aa_list]

        for ind, aa_seq in enumerate(aa_list):
            for j, aa in enumerate(aa_seq):
                if j > max_aa - 1:
                    break
                x_data[ind][int(aa_map[aa]), j] = 1

    data_set = x_data

    return data_set


def dream_parser(filename, threshold):

    # if filename[-2] == 'gz':

    f = gzip.open(filename, 'rb')

    x_data = []

    y_data = []

    for line in f:
        aa_seq, dnase, label, mask = line.split('\t')
        aa_seq = np.concatenate([nt_map[i] for i in aa_seq]).astype(np.float32)
        dnase = np.asarray(eval(dnase), dtype=np.float32)
        label = np.asarray(eval(label), dtype=np.float32)
        np.place(label, label == 2., threshold)
        mask = np.asarray(eval(mask), dtype=np.float32)

        x_data.append(np.concatenate((aa_seq, dnase)))
        y_data.append(np.concatenate((label, mask)))

    return np.asarray(x_data), np.asarray(y_data)



