# coding=utf-8
try:
    import cPickle
except ImportError:
    import pickle as cPickle
import numpy as np
import pandas as pd
from Bio import SeqIO

from .seq_tools import aa2codon, hot2aa, nt_map, secs2hot


def type2p_code(description):
    """
        This module parses data from REBASE and transforms them for Evolutron
    """
    # Read fasta input
    rec_site = description.split(' ')[-1]

    code = np.zeros((len(rec_site), 4), dtype=np.float32)
    for j, nt in enumerate(rec_site):
        code[j, :] = np.asarray(nt_map[nt])

    return code.flatten()


def fasta_parser(filename, codes=False, code_key=None):
    """
        This module parses data from FASTA files and transforms them to Evolutron format.
    """
    input_file = open(filename, "rU")

    aa_list = []
    code_list = []
    for record in SeqIO.parse(input_file, "fasta"):
        aa_list.append(str(record.seq))
        if codes:
            try:
                if code_key == 'scop':
                    code_list.append(record.description.split('|')[-1])
                elif code_key == 'type2p':
                    code_list.append(type2p_code(record.description))
                else:
                    raise IOError('You had codes=True but did not provide a code_key')
            except:
                raise IOError('Fasta parser code option set to True, but code was not recognized')

    raw_data = pd.DataFrame()
    raw_data['sequence'] = pd.Series(aa_list)

    x_data = raw_data.sequence

    if codes:
        y_data = code_list
    else:
        y_data = None

    return x_data, y_data


def csv_parser(filename, codes=False, code_key=None, sep='\t'):
    def fam(x):
        dt = str(x).split(',')
        f = [d for d in dt if d.find(' family') >= 0]

        try:
            return f[0]
        except IndexError:
            return "Unassigned"

    def supfam(x):
        dt = str(x).split(',')
        f = [d for d in dt if d.find('superfamily') >= 0]

        try:
            return f[0]
        except IndexError:
            return "Unassigned"

    def subfam(x):
        dt = str(x).split(',')
        f = [d for d in dt if d.find('subfamily') >= 0]

        try:
            return f[0]
        except IndexError:
            return "Unassigned"

    try:
        raw_data = pd.read_hdf(filename.split('.')[0] + '.h5', 'raw_data')
    except FileNotFoundError:
        raw_data = pd.read_csv(filename, sep=sep, header='infer')
        raw_data.columns = raw_data.columns.str.strip().str.lower().str.replace(' ', '_')
        try:
            raw_data['fam'] = raw_data['protein_families'].apply(fam)
            raw_data['sup'] = raw_data['protein_families'].apply(supfam)
            raw_data['sub'] = raw_data['protein_families'].apply(subfam)
        except KeyError:
            pass

        raw_data.to_hdf(filename.split('.')[0] + '.h5', 'raw_data')

    if codes:
        if type(code_key) == str:
            code_vc = raw_data[code_key].value_counts()
            pos_data = raw_data[raw_data[code_key] != 'Unassigned'][
                ~raw_data[code_key].isin(code_vc[code_vc == 1].index.tolist())]
            code_cats = pos_data[code_key].astype('category').cat.codes
            y_data = code_cats.tolist()
            y_data = [y + 1 for y in y_data]
            x_data = pos_data.sequence
        else:
            pf = raw_data[code_key]
            pos_data = raw_data[pf[code_key[0]].notnull() & pf[code_key[1]].notnull()]
            y_data = [pos_data[k].tolist() for k in code_key]
            x_data = pos_data.sequence
    else:
        x_data = raw_data.sequence
        y_data = None

    return x_data, y_data


def h5_parser(filename, key='raw_data', x_codes='sequence', y_codes=None):
    raw_data = pd.read_hdf(filename, key)

    if y_codes:
        if type(y_codes) == str:
            pos_data = raw_data[raw_data[y_codes].notnull()]
            y_data = pos_data[y_codes].values
        else:
            pf = raw_data[y_codes]
            pos_data = raw_data[pf[y_codes[0]].notnull() & pf[y_codes[1]].notnull()]
            y_data = [pos_data[code].values for code in y_codes]
    else:
        pos_data = raw_data
        y_data = None

    if type(x_codes) == str:
        x_data = pos_data[x_codes]
    else:
        x_data = [pos_data[code] for code in x_codes]

    return x_data, y_data


def secs_parser(filename, nb_categories=8, nb_aa=20, dummy_option=None):
    """
        This module parses data from files containing sequence and secondary structure
        and transforms them to Evolutron format.
    """

    input_file = open(filename, "rU")

    aa_list = []
    secs_list = []
    flag = True
    for record in SeqIO.parse(input_file, "fasta"):
        if flag:
            seq = str(record.seq)
            aa_list.append(seq)

            flag = False
        else:
            sec = str(record.seq)
            secs_list.append(sec)

            flag = True

    raw_data = pd.DataFrame()
    raw_data['sequence'] = pd.Series(aa_list)
    raw_data['secs'] = pd.Series(secs_list)

    x_data = raw_data.sequence

    y_data = list(map(lambda x: secs2hot(x, nb_categories), secs_list))

    return x_data, y_data


def npz_parser(filename, nb_categories=8, pssm=False, codon_table=False,
               extra_features=False, nb_aa=22, dummy_option=None):
    """
        This module parses data from npz files containing sequence and secondary structure
        and transforms them to Evolutron format.
    """

    data = np.load(filename)

    data = np.reshape(data[:], (-1, 700, 57))

    x_data = data[:, :, :nb_aa]

    if pssm:
        x_data = np.concatenate((x_data, data[:, :, 35:35 + nb_aa]), axis=-1)
    if extra_features:
        x_data = np.concatenate((x_data, data[:, :, 31:33]), axis=-1)
    if codon_table:
        codons = []
        for i in range(data.shape[0]):
            codons.append(aa2codon(hot2aa(data[i, :, 0:22])))
        x_data = np.concatenate((x_data, codons), axis=-1)

    y_data = data[:, :, 22:30]

    return x_data, y_data
