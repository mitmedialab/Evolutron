from __future__ import print_function

from rosetta import *
# from random import randint, random

# import sys
# import time

import lasagne
import numpy as np
import theano
import theano.tensor as ten
# import prep_v031 as prep
from predict_type2p import build_network
from toolbox import get_secstruct

from toolbox import generate_resfile_from_pose


# noinspection PyShadowingNames
def load_evolutron(filename):
    # Prepare Theano variables for inputs and targets as well as index to minibatch
    inputs = ten.tensor3('inputs')

    # Create neural network model (parametrization based command line inputs or else defaults)
    print("Building model and compiling functions...")

    with np.load(filename) as f:
        [filters, filter_size] = f['arr_0'][2:]
        param_values = [f['arr_%d' % i] for i in range(1, len(f.files))]

    network = build_network(inputs, 700, filters, filter_size)

    lasagne.layers.set_all_param_values(network, param_values)

    # Create a loss expression for training, i.e., a scalar objective we want to minimize
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_pred = theano.function([inputs], test_prediction)

    return test_pred


# # set up mover
def perturb_bb(pose):
    resnum = randint(14, pose.total_residue())
    pose.set_phi(resnum, pose.phi(resnum) - 25 + random() * 50)
    pose.set_psi(resnum, pose.psi(resnum) - 25 + random() * 50)
    return pose


def main():
    # Initialize rosetta
    init()

    # Initialize Rosetta scoring function
    score_ros = get_fa_scorefxn()
    # scorefxn2 = ScoreFunction()
    # scorefxn2.set_weight(fa_atr, 1.0)
    # scorefxn2.set_weight(fa_rep, 1.0)

    # Initialize Evolutron scoring function
    filename = 'models/o_smodel_500_150_30_30.npz'
    score_cnn = load_evolutron(filename)

    # Load initial protein and view in PyMol
    pose = Pose()
    pose_from_file(pose, 'structures/eco_dna.pdb')

    pymol = PyMOL_Mover()
    pymol.apply(pose)

    # set up MonteCarlo object
    kT = 1.0
    mc = MonteCarlo(pose, score_ros, kT)

    # Pack Rotamers Mover
    generate_resfile_from_pose(pose, "1eri.resfile")
    task_design = TaskFactory.create_packer_task(pose)
    parse_resfile(pose, task_design, "1eri.resfile")

    pack_mover = PackRotamersMover(score_ros, packer_task)
    pack_mover.apply(pose)


    # Monte-Carlo search
    mc.reset(pose)  # assumes we have the lowest energy structure
    for i in range(1, 60000):
        perturb_bb(pose)  # make a change (Search Operator, Mover in Rosetta)
        mc.boltzmann(
            pose)  # calculating Boltzmann energy and evaluating with Metropolis criterion, keep changes if
        # exp(-DE/kT) < than before

        if i % 1000 == 0:
            mc.recover_low(pose)
    # output lowest-energy structure
    mc.recover_low(pose)

    # GA Search

    # Start with the initial (native) pose (protein structure)

    # for i in range(1, n_generations):
        #if first gen:
            # Perform mutations/change of angles +++ on the initial pose
            # and store generated population
            # Calculate energy scores for members of population (fitness function)
            # Perform selection and select a curated subset of individuals (poses)
            # Let's keep X best individuals based on running time
        #else:
            # Perform mutations/crossover/change of angles +++ on the  selected individuals
            # and store generated population
            # Calculate energy scores for members of population (fitness function)
            # Perform selection and select a curated subset of individuals (poses)
            # Let's keep X best individuals based on running time