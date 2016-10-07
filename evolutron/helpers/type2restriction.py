#!/usr/bin/env python
#
#      Type 2 Restriction Enzymes Learning Software.
#      Copyright (C) 2015. Thrasyvoulos Karydis.
#      Massachusetts Institute of Technology
#
"""
This file assumes that the rebase files exist in datasets, and builds up the entire type 2 restriction enzymes
database needed for the rest of the programs
Import as : import type2restriction
"""

import re
from collections import defaultdict

from Bio import SeqIO
from Bio.Alphabet import DNAAlphabet, ProteinAlphabet
from Bio.Seq import Seq


############################
# -------- classes ------- #
############################
class Protein(object):
    def __init__(self, name, aa_seq, nt_seq):
        self.name = name
        self.aa_seq = aa_seq
        self.nt_seq = nt_seq
        self.aa = str(self.aa_seq)
        self.nt = str(self.nt_seq)
        self.aa_num = len(aa_seq)
        self.nt_num = len(nt_seq)


class RestrictionEnzyme(Protein):
    def __init__(self, name, aa_seq, nt_seq, rec_site):
        """

        :type rec_site: Nucleotide sequence of the recognition site
        """
        Protein.__init__(self, name, aa_seq, nt_seq)
        self.rec_site = rec_site
        if name[-1] == 'P':
            self.is_pseudo = True
        else:
            self.is_pseudo = False

    def __str__(self):
        return '{x.name} {x.rec_site} {x.nt_num} nt {x.aa_num} aa'.format(x=self)


class RecognitionSite:
    def __init__(self, sequence, iso, iso_p):
        self.seq = Seq(sequence, DNAAlphabet())
        self.iso = iso
        self.isoP = iso_p
        self.sequence = sequence

        if self.seq.reverse_complement() == self.seq:
            self.is_type2p = True
        else:
            self.is_type2p = False

        if re.match('[ACGT]*$', str(self.seq)):
            self.is_ambig = False
        else:
            self.is_ambig = True

    def __str__(self):
        return self.sequence

    def count_cutters(self):
        print '{x.iso} true, {x.isoP} pseudo'.format(len(self.iso), len(self.isoP))


########################################
# ------ Construction Functions ------ #
########################################
def enzymes_dict():
    enzymes = dict()
    aa_iterator = SeqIO.parse(open("datasets/type2_rebase_aa.fasta", 'r'), 'fasta')
    nt_iterator = SeqIO.parse(open("datasets/type2_rebase_nt.fasta", 'r'), 'fasta')
    for aa, nt in zip(aa_iterator, nt_iterator):
        aa.seq.alphabet = ProteinAlphabet()
        nt.seq.alphabet = DNAAlphabet()
        desc = filter(None, aa.description.split(' '))
        if desc[1].isdigit():
            rec_site = None
        else:
            rec_site = desc[1]
        if 'fragment' not in desc:
            enzymes[aa.name] = RestrictionEnzyme(aa.name, aa.seq, nt.seq, rec_site)

    return enzymes


def sites_dict():
    sites = dict()
    temp = defaultdict(list)

    for re in type2re.values():
        if re.rec_site:
            temp[re.rec_site].append(re.name)

    for t in temp:
        t_iso = []
        t_iso_p = []
        for name in temp[t]:
            if name[-1] != 'P':
                t_iso.append(name)
            else:
                t_iso_p.append(name)
        if len(t_iso) > 0:
            sites[t] = RecognitionSite(t, t_iso, t_iso_p)

    return sites


########################################
# ------- Dynamic Construction ------- #
########################################

type2re = enzymes_dict()  # Name: {RestrictionEnzyme}
type2site = sites_dict()  # Site: {RecognitionSite}

type2res = type2re.values()
type2sites = type2site.values()
