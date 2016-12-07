import os

prosite_res = [x.strip('\n') for x in open('prosite_c2h2.results', 'r').readlines()]

# Hits for PS50157 (ZINC_FINGER_C2H2_2) [PROSITE (release 20.128)] motif on all UniProtKB/Swiss-Prot
# (release 2016_07 of 06-Jul-16: 551705 entries) database sequences :

files = ['outputs/CoMET.search/' + f for f in os.listdir('outputs/CoMET.search/') if f[-2:] == '.o' and 'b1h' in f]

csv_file = open('b1h_results.csv', 'w')

csv_file.write('filters,filter_size,CoMET,common,extra,missed\n')

for filename in files:
    lines = open(filename, 'r').readlines()

    missed_file = open(filename[:-2] + '.missed', 'w')

    extra_file = open(filename[:-2] + '.extra', 'w')

    comet_res = []
    for l in lines:
        comet_res.append(l.split('|')[1])
        if l.split('|')[1] not in prosite_res:
            extra_file.write(l)

    common = set(prosite_res) & set(comet_res)

    union = set(prosite_res) | set(comet_res)

    comet_only = set(comet_res) - set(prosite_res)

    prosite_only = set(prosite_res) - set(comet_res)
    missed_file.write('\n'.join(list(prosite_only)))

    filters = filename.split('_')[2]
    filter_size = filename.split('_')[3].split('.')[0]

    # TODO: add classification here

    csv_file.write('{0},{1},{2},{3},{4},{5}\n'.format(filters, filter_size, len(comet_res), len(common),
                                                      len(comet_only), len(prosite_only)))

csv_file.close()
