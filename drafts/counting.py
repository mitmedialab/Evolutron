from itertools import product

from Bio import Seq

import type2restriction as t2re

import itertools


def extend_ambiguous_dna(seq):
    """return list of all possible sequences given an ambiguous DNA input"""
    d = Seq.IUPAC.IUPACData.ambiguous_dna_values
    return list(map("".join, product(*map(d.get, seq))))


dna_amb = ['N', 'M', 'R', 'W', 'S', 'Y', 'K', 'V', 'H', 'D', 'B']

sites = []

for i in xrange(8):
    sites.append([t for t in t2re.type2site if len(t) == i + 1 and sum(map(t.find, dna_amb)) == -len(dna_amb)])

sites_a = []
for i in xrange(8):
    sites_a.append([extend_ambiguous_dna(t) for t in t2re.type2site if
                    len(t) == i + 1 and sum(map(t.find, dna_amb)) != -len(dna_amb)])

for i in xrange(8):
    sites_a[i] = list(itertools.chain(*sites_a[i]))

print("Data for specific proteins")
print('-' * 60)
for i in xrange(8):
    print("Rec. sites with length {0}: {1}".format(i + 1, len(sites[i])))

print("Data with ambiguous rec. site proteins")
print('-' * 60)
for i in xrange(8):
    print("Rec. sites with length {0}: {1}".format(i + 1, len(sites_a[i])))
