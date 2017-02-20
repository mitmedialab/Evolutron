# coding=utf-8
import numpy as np

from evolutron.tools import io_tools as io

file_db = {
    'random': '/data/datasets/random_aa.fasta',
    'type2p': '/data/datasets/type2p_ps_aa.fasta',
    'type2': '/data/datasets/sprot_type2_pfam.tsv',
    'hsapiens': '/data/datasets/sprot_hsapiens_pfam.tsv',
    'ecoli': '/data/datasets/sprot_ecoli_pfam.tsv',
    'zinc': '/data/datasets/sprot_znf_prot_pfam.tsv',
    'homeo': '/data/datasets/sprot_homeo_pfam.tsv',
    'crispr': '/data/datasets/sprot_crispr_pfam.tsv',
    'cas9': '/data/datasets/sprot_cas9_pfam.tsv',
    'cd4': '/data/datasets/sprot_cd4_pfam.tsv',
    'dnabind': '/data/datasets/sprot_dna_tf_pfam.tsv',
    'SecS': '/data/datasets/SecS.sec',
    'smallSecS': '/data/datasets/smallSecS.sec',
    'tinySecS': '/data/datasets/tinySecS.sec',
    'cullPDB': '/data/datasets/cullpdb+profile_6133_filtered.npy.gz',
    'cb513': '/data/datasets/cb513+profile_split1.npy.gz',
    'human_ors': '/data/datasets/uniprot_human_ors.tsv',
    'casp10': '/data/datasets/casp10.sec',
    'casp11': '/data/datasets/casp11.sec',
    'hsapx': '/data/datasets/sprot_hsapiens_expr_pfam.tsv',
    'scop': '/data/datasets/scop2.fasta',
    'swissprot': '/data/datasets/sprot_all_pfam.tsv',
    'acetyl': '/data/datasets/sprot_ec2_3_pfam.tsv',
    'small_all': '/data/datasets/small_uniprot-all.tsv',
    'mycoplasma': '/data/datasets/uniprot_mycoplasma_pfam.tsv',
    'small_all': '/data/datasets/small_uniprot-all.tsv',
    'go': '/data/datasets/sprot_go.tsv'
}


def data_it(dataset, block_size):
    """ Iterates through a large array, yielding chunks of block_size.
    """
    size = len(dataset)

    for start_idx in range(0, size, block_size):
        excerpt = slice(start_idx, min(start_idx + block_size, size))
        yield dataset[excerpt]


def pad_or_clip_seq(x, n):
    if n >= x.shape[0]:
        return np.concatenate((x[:n, :], np.zeros((n - x.shape[0], x.shape[1]))))
    else:
        return x[:n, :]


def load_dataset(data_id, padded=True, min_aa=None, max_aa=None, pad_y_data=False, **parser_options):
    """Fetches the correct dataset from database based on data_id.
    """
    try:
        filename = file_db[data_id]
        filetype = filename.split('.')[-1]
    except KeyError:
        raise IOError('Dataset id not in file database.')

    if filetype == 'tsv':
        x_data, y_data = io.tab_parser(filename, **parser_options)
    elif filetype == 'fasta':
        x_data, y_data = io.fasta_parser(filename, **parser_options)
    elif filetype == 'sec':
        x_data, y_data = io.SecS_parser(filename, **parser_options)
    elif filetype == 'gz':
        x_data, y_data = io.npz_parser(filename, **parser_options)
    else:
        raise NotImplementedError('There is no parser for current file type.')

    if padded:
        if not max_aa:
            max_aa = int(np.percentile([len(x) for x in x_data], 99))  # pad so that 99% of datapoints are complete
        else:
            max_aa = min(max_aa, np.max([len(x) for x in x_data]))

        x_data = np.asarray([pad_or_clip_seq(x, max_aa) for x in x_data])

        if min_aa:
            min_aa = max(min_aa, np.max([len(x) for x in x_data]))
            x_data = np.asarray([pad_or_clip_seq(x, min_aa) for x in x_data])

        if pad_y_data:
            try:
                y_data = np.asarray([pad_or_clip_seq(y, min_aa) for y in y_data])
                return x_data, y_data
            except:
                pass

    data_size = len(x_data)

    # ToDo: The old format returns an error when y_data is an array. CHECK!
    #if not y_data:
    if y_data is None:
        # Unsupervised Learning
        # x_data: observations

        print('Dataset size: {0}'.format(data_size))
        return x_data
    else:
        # Supervised Learning
        # x_data: observations
        # y_data: class labels
        try:
            assert ((len(x_data) == len(y_data)) or (len(x_data) == len(y_data[0])))
        except AssertionError:
            raise IOError('Unequal lengths for X ({0}) and y ({1})'.format(len(x_data), len(y_data[0])))

        print('Dataset size: {0}'.format(data_size))
        return x_data, np.asarray(y_data)

