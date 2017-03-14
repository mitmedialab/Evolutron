import os
import pandas as pd

os.chdir('/data/datasets')

ppi_id_id = pd.read_csv('ppi.tsv', sep=' ')

id_uniq = pd.concat([ppi_id_id.ID1, ppi_id_id.ID2]).unique().tolist()

# Write the unique ids
f = open('ppi_unique_id', 'w')
f.writelines("%s\n" % i for i in id_uniq)
f.close()

# Submit them to uniprot

# get the sequences
ppi_id_seq = pd.read_csv('uniprot_ppi.tsv', sep='\t')

# No sequences for
no_seq = list(set(id_uniq) - set(ppi_id_seq.Entry.tolist()))

# Got the extra sequences, redundancy to 1st column

ppi_id_seq_extra = pd.read_csv('uniprot_ppi_extra.tsv', sep='\t')

ppi_id_seq_extra_fixed = pd.DataFrame(columns=ppi_id_seq_extra.columns)
count = 0
for index, row in ppi_id_seq_extra.iterrows():
    ids = row['yourlist:M20170314F725F458AC8690F874DD868E4ED79B88A8FA155']
    for i in ids.split(','):
        x = row
        row['yourlist:M20170314F725F458AC8690F874DD868E4ED79B88A8FA155'] = i
        print(row)
        print(type(row))
        ppi_id_seq_extra_fixed.loc[count] = row
        count += 1

ppi_id_seq_extra_fixed['Entry'] = ppi_id_seq_extra_fixed['yourlist:M20170314F725F458AC8690F874DD868E4ED79B88A8FA155']

# At last get final list with all the proteins
ppi_id_seq = ppi_id_seq.append(ppi_id_seq_extra_fixed)
ppi_id_seq.index = ppi_id_seq['Entry']

no_seq_2 = list(set(id_uniq) - set(ppi_id_seq.Entry.tolist()))

ppi_id_seq_dict = ppi_id_seq[['Entry', 'Sequence']].to_dict()['Sequence']

ppi_id_id['ID1'] = ppi_id_id['ID1'].apply(lambda i: ppi_id_seq_dict[i])
ppi_id_id['ID2'] = ppi_id_id['ID2'].apply(lambda i: ppi_id_seq_dict[i])

ppi_id_id.to_hdf('ppi_seq.h5', key='raw_data')
