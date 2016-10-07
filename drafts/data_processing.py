import gzip
import glob
from itertools import izip

all_reg = open('small_allregion.seq', 'rU')
keep_reg = open('small_keepregion.list', 'rU')
all_dnase = [gzip.open(f, 'rU') for f in glob.iglob('small_raw/*.dnase.gz')]
all_labels = [gzip.open(f, 'rU') for f in glob.iglob('small_raw/*.label.gz')]
kept_dnase = [open('small_kept/'+f.split('/')[1][:-3], 'w') for f in glob.iglob('small_raw/*.dnase.gz')]
kept_labels = [open('small_kept/'+f.split('/')[1][:-3], 'w') for f in glob.iglob('small_raw/*.label.gz')]


kept_reg = open('small_validregion.seq', 'w')

keep_index = keep_reg.readline().strip().split()[3]
count = 0
for line in all_reg:
    dnase_data = []
    labels_data = []
    for f in all_dnase:
        dnase_data.append(f.readline())
    for f in all_labels:
        labels_data.append(f.readline())
    count += 1
    if int(keep_index) == count:
        kept_reg.write(line)
        keep_index = keep_reg.readline().strip().split()[3]
        for f, l in izip(kept_dnase, dnase_data):
            f.write(l)
        for f, l in izip(kept_labels, labels_data):
            f.write(l)

print(count)
print(keep_index)

