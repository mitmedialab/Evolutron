"""
This program scores the mutations of the contact residues of an aa sequence based on the prediction of Evolutron.
"""
from __future__ import print_function

import os
import random

import corebio
import lasagne
import numpy as np
import theano
import theano.tensor as ten
from weblogolib import *

import prep_v031 as prep
import type2restriction as t2re
from predict_type2p import build_network


def load_evolutron(filename):
    # Prepare Theano variables for inputs and targets as well as index to minibatch
    inputs = ten.tensor3('inputs')

    # Create neural network model (parametrization based command line inputs or else defaults)
    print("Building model and compiling functions...")

    with np.load(filename) as f:
        [filters, filter_size] = f['arr_0'][2:]
        param_values = [f['arr_%d' % i] for i in range(1, len(f.files))]

    network = build_network(inputs, 400, filters, filter_size)

    lasagne.layers.set_all_param_values(network, param_values)

    # Create a loss expression for training, i.e., a scalar objective we want to minimize
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_pred = theano.function([inputs], test_prediction)

    return test_pred


def mutate1_sequence(seq, positions):
    mutated = []
    names = []
    for p in positions:
        for aa in prep.aa_let:
            mutated.append(seq[:p - 1] + aa + seq[p:])
            names.append('{0}{1}{2}'.format(seq[int(p)], p, aa))

    return mutated, names


def id_generator(size=6, chars=prep.aa_let):
    return ''.join(random.choice(chars) for _ in range(size))


def mutation_sample(seq, positions, size):
    mutated = []
    names = []
    for i in xrange(size):
        mut_seq = list(seq)
        name = ''
        mutations = id_generator(len(positions))
        for j, p in enumerate(positions):
            mut_seq[int(p)] = mutations[j]
            name += '{0}{1}{2}'.format(seq[int(p)], p, mutations[j])
        mutated.append(''.join(mut_seq))
        names.append(name)

    return mutated, names


# Load the network to use for the evaluations
net_file = 'models/o_smodel_500_150_100_6.npz'
score_cnn = load_evolutron(net_file)

# Load sequence to be tested
n_aa = 700

ecori = str(t2re.type2re['EcoRI'].aa_seq)

ecori_oh = np.zeros((20, n_aa), dtype=np.float32)
for j, aa in enumerate(ecori):
    if j < n_aa:
        ecori_oh[int(prep.aa_map[aa]), j] = 1

ecori_site = t2re.type2re['EcoRI'].rec_site

ecori_site_oh = np.zeros((6, 4), dtype=np.float32)
for j, nt in enumerate(ecori_site):
    ecori_site_oh[j, :] = np.asarray(prep.nt_map[nt])

# Get contact residues  (should write this, now hardcoded)
con_res = [62, 85, 87, 89, 91, 113, 117, 138, 145, 148]

con_res2 = [random.choice(map(str, range(200))) for _ in range(30)]

# Generate mutations in contact residues
# ecori_muts, names = mutate1_sequence(ecori, con_res)
ecori_muts, names = mutation_sample(ecori, con_res, 10000)

ecori_muts_oh = np.zeros((len(ecori_muts), 20, n_aa), dtype=np.float32)

for i, aa_seq in enumerate(ecori_muts):
    for j, aa in enumerate(aa_seq):
        if j < n_aa:
            ecori_muts_oh[i, int(prep.aa_map[aa]), j] = 1

y_pred = [score_cnn([i]) for i in ecori_muts_oh]
y_pred.append(score_cnn([ecori_oh]))
y_pred = np.asarray(y_pred)
y_pred = y_pred.reshape((-1, 6, 4))
scores = [((y_pred[i] - ecori_site_oh) ** 2).mean() for i in xrange(len(y_pred))]


# np.savez_compressed('outputs/sensitivity/ecori/' + net_file[9:] + '.preds', mutants=ecori_muts,
#                     y_pred=y_pred)

print("done")

# this_test_loss = np.mean(test_losses)

for k, pred in enumerate(y_pred):
    for i in xrange(len(pred)):
        y_pred[k, i] = (pred[i] - min(pred[i])) / (max(pred[i]) - min(pred[i]))
        y_pred[k, i] /= sum(pred[i])

print('Started visualizations')
foldername = 'outputs/sensitivity/ecori/' + net_file[9:] + '.preds'
if not os.path.isdir(foldername):
    os.mkdir(foldername)

print(len(y_pred))

for i, pred_motif in enumerate(y_pred):
    if scores[i] > np.mean(scores) + 3*np.std(scores):
        seq_name = names[i]  # write mut's here
        if not os.path.isfile(foldername + '/' + seq_name + '.pdf'):
            data = LogoData.from_counts(corebio.seq_io.unambiguous_dna_alphabet, pred_motif)
            options = LogoOptions()
            options.title = seq_name
            my_format = LogoFormat(data, options)
            my_pdf = pdf_formatter(data, my_format)
            foo = open(foldername + '/' + seq_name + '.pdf', "w")
            foo.write(my_pdf)
            foo.close()
    if i == len(y_pred) - 1:
        seq_name = 'native_pred'
        if not os.path.isfile(foldername + '/' + seq_name + '.pdf'):
            data = LogoData.from_counts(corebio.seq_io.unambiguous_dna_alphabet, pred_motif)
            options = LogoOptions()
            options.title = seq_name
            my_format = LogoFormat(data, options)
            my_pdf = pdf_formatter(data, my_format)
            foo = open(foldername + '/' + seq_name + '.pdf', "w")
            foo.write(my_pdf)
            foo.close()

seq_name = 'native_test'  # write mut's here
data = LogoData.from_counts(corebio.seq_io.unambiguous_dna_alphabet, ecori_site_oh)
options = LogoOptions()
options.title = seq_name
my_format = LogoFormat(data, options)
my_pdf = pdf_formatter(data, my_format)
foo = open(foldername + '/' + seq_name + '.pdf', "w")
foo.write(my_pdf)
foo.close()


print('Output available')
