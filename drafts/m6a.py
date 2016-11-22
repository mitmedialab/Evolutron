from Bio import Entrez
from Bio import SeqIO

import csv



def get_aa_seqs():  # Getting bulk sequence data from NCBI
    Entrez.email = "karydis@media.mit.edu"
    infile = 'datasets/m6a_aa.csv'
    outfile = 'datasets/m6a.csv'

    data = csv.DictReader(open(infile, 'r'))

    csvwriter = csv.DictWriter(open(outfile, 'a'), data.fieldnames)

    csvwriter.writeheader()

    data = list(data)

    for i, d in enumerate(data[19137:]):
        print i, d['gbid']
        if not d['gbid'] == "['']":
            handle = Entrez.efetch(db="nuccore",
                                   id=d['gbid'],
                                   rettype="gb",
                                   retmode='text')
            gbdata = SeqIO.read(handle, "genbank")
            aa_seq = [x.qualifiers['translation'] for x in gbdata.features if x.type == 'CDS']
        else:
            aa_seq = ['']

        d['AA'] = aa_seq
        csvwriter.writerow(dict(d))
