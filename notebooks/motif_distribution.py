import matplotlib.pyplot as plt
import numpy as np
import os
from Bio import SeqIO

if 'drafts' in os.listdir(os.curdir):
    os.chdir('drafts/')
if not os.path.exists('lengths.npz'):
    lengths = []
    for f in [g for g in os.listdir('.') if 'msa' in g]:
        seq = [s for s in SeqIO.parse(open(f, 'r'), "fasta")][0]

        lengths.append(len(seq))
    np.savez('lengths', lengths=np.asarray(lengths))
else:
    data = np.load('lengths.npz')
    lengths = data['lengths']

n, bins, patches = plt.hist(lengths, 40, range=(0,200), facecolor='green', alpha=0.75)
plt.xlim([0, 200])
plt.xlabel('Motif size')
plt.ylabel('# of Motifs')
plt.title('Size Distribution of PROSITE Motifs')
