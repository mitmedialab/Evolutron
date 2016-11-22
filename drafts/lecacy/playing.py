from Bio import SeqIO
import cPickle
# first i am gonna find all the 1-mers

# Found: one_mers = ['M', 'A', 'F', 'S', 'E', 'D', 'V', 'L', 'K', 'Y', 'R', 'P', 'N', 'W', 'Q', 'C', 'G', 'I', 'H', 'T', 'X', 'Z', 'B', 'U', 'O']

handle = open("datasets/uniprot_sprot.fasta", "rU")

four_mers = []
c = 0
for record in SeqIO.parse(handle, "fasta"):

    s = record.seq.__str__()

    for i in xrange(len(s)):
        if s[i:i + 4] not in four_mers:
            four_mers.append(s[i:i + 4])
    c += 1
    if c % 100000 == 0:
        print(c)

cPickle.dump(four_mers, open('fourmers','wb'))

handle.close()
