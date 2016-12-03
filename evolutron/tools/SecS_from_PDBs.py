from Bio import PDB

import glob

pdb_files = glob.iglob('all_pdbs/*')

file = open('casp11.sec', 'w')

c=0
for pdb in pdb_files:
        c+=1
        print(c)

        p = PDB.PDBParser()
        structure = p.get_structure(pdb[:-4], pdb)
        model = structure[0]
        dssp = PDB.DSSP(model, pdb)

        seq = ''
        ss = ''
        for key in list(dssp.keys()):
            ss += dssp[key][2]
            seq += dssp[key][1]

        file.write('>{}\n'.format(pdb))
        file.write('{}\n'.format(seq))
        file.write('>{}\n'.format(pdb))
        file.write('{}\n'.format(ss))


file.close()

