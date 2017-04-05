from __future__ import print_function

import os
import random
from optparse import OptionParser

import Bio.PDB
from Bio.PDB import *
from Bio.PDB import PDBIO
from Bio.PDB import PDBParser


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_atom_list(structure, chains):
    output = dict()
    for chain in structure:
        if chain.id in chains:
            for residue in chain.get_residues():
                hetflag, resseq, icode = residue.get_id()
                the_id = (chain.id + "_" + str(resseq) + "_" + icode).strip()
                for atom in residue.get_unpacked_list():
                    if hetflag == ' ':
                        if the_id in output:
                            output[the_id].append(atom)
                        else:
                            output[the_id] = [atom]
    return output


def is_contact(res_1, other_atoms, cutoff):
    for atom in res_1:
        ns = NeighborSearch(other_atoms)
        center = atom.get_coord()
        neighbors = ns.search(center, cutoff)  # 5.0 for distance in angstrom
        residue_list = Selection.unfold_entities(neighbors, 'R')  # R for residues
        if len(residue_list) > 0:
            return True
    return False


def get_contacts(struc, all_atoms, verbose, cutoff):
    progress = 0
    contacts = []
    for residue in struc:
        progress += 1
        if len(verbose) > 0:
            print(verbose, progress, "out of", len(struc))
        atom_list = struc[residue]
        outcome = is_contact(atom_list, all_atoms, cutoff)
        if outcome:
            contacts.append(residue)
    return contacts


# Filter out all the atoms from the chain,residue map given by residue_map
def get_all_atoms(residue_map):
    all_atoms_out = []
    for residue in residue_map:
        for atom in residue_map[residue]:
            all_atoms_out.append(atom)
            # Set the b-factor to zero for coloring by contacts
            atom.set_bfactor(0.0)
    return all_atoms_out


# Save the structures with B-factor field replaced with contact (100) and interface neighborhood (50)
def save_contacts(structure, chains, out_file):
    # Save only those chains that we are supposed to
    Select = Bio.PDB.Select

    class ConstrSelect(Select):
        def accept_chain(self, chain):
            # print dir(residue)

            if chain.id in chains:
                return 1
            else:
                return 0

    w = PDBIO()
    w.set_structure(structure)
    randint = random.randint(0, 9999999)
    w.save("TMP" + str(randint) + ".pdb", ConstrSelect())
    # Remove the HETATM and TER lines
    f_tmp = open("TMP" + str(randint) + ".pdb", 'r')
    f_out = open(out_file, 'w')
    for line in f_tmp.readlines():
        if line[0:3] != "TER" and line[0:6] != "HETATM":
            f_out.write(line)
    f_tmp.close()
    f_out.close()
    os.remove("TMP" + str(randint) + ".pdb")


# Save the residues which are contacts or neighborhood interface in a space-delimited file
def save_residues(filename, interface, contacts):
    f = open(filename, 'w')
    for elem in interface:
        splitted = elem.split("_")
        resname = str((splitted[1] + splitted[2]).strip())
        chain = splitted[0]
        # contact or neighbor of interface?
        coninf = "I"  # Interface neighbor
        if elem in contacts:
            coninf = "C"  # Contact
        f.write(chain + " " + resname + " " + coninf + "\n")
    f.close()


# only save the residues of the file that are in thelist
def save_constrained(filename_in, filename_out, thelist):
    f = open(filename_in, 'r')
    f_out = open(filename_out, 'w')
    for line in f.readlines():
        if "ATOM" in line:
            line = line.strip()
            resname = line[23:28].strip()
            icode = ""
            if (not is_number(resname[len(resname) - 1])):
                icode = resname[len(resname) - 1]
                resname = resname[0:(len(resname) - 1)]

            resname = line[21] + "_" + resname + "_" + icode
            if resname in thelist:
                f_out.write(line + "\n")


# Parse the options
usage = "USAGE: python GetContacts.py --f1 FirstPDB --f2 SeconPDB --c1 FirstPDBChains --c2 SecondMoleculeChains [--c ContactCutoff] [--i InterfaceCutoff] [--o1 output1] [--o2 output2] \n"
parser = OptionParser(usage=usage)

# options
parser.add_option("--f1", help="First molecule pdb", dest="f1")
parser.add_option("--f2", help="Second molecule pdb", dest="f2")
parser.add_option("--c1", help="First molecule chains", dest="c1")
parser.add_option("--c2", help="Second molecule chains", dest="c2")
parser.add_option("--c", help="Distance cutoff between contacting residues", dest="cutoff")
parser.add_option("--i", help="Interface distance cutoff between contacting residues", dest="interfacial")
parser.add_option("--jobid", help="Job name for the output folder", dest="jobid")

(options, args) = parser.parse_args()
cutoff = 4.5
i_cutoff = 10.0
out_folder = ""
if (options.f1 and options.f1 and options.c1 and options.c2):
    if (options.cutoff):
        cutoff = float(options.cutoff)
    else:
        print("No contact cutoff supplied, using default of 4.5A")
    if (options.interfacial):
        i_cutoff = float(options.interfacial)
    else:
        print("No interfacial cutoff supplied, using default of 10.0A")
    if (options.jobid):
        out_folder = "out_" + options.jobid
    else:
        out_folder = "out_" + str(random.randint(0, 999999))
        print("No job_id supplied, creating a folder with a random name: " + out_folder)
else:
    print("Not enough input arguments supplied")
    print(usage)
    quit()

cwd = os.getcwd()
os.system("mkdir " + cwd + "/" + out_folder)

str_1 = PDBParser().get_structure('first_one', options.f1)  # load your molecule
str_2 = PDBParser().get_structure('second_one', options.f2)  # load your molecule

chains_1 = options.c1

chains_2 = options.c2

# Load the structures - they can be the same!
atoms_1 = Selection.unfold_entities(str_1, 'C')  # C for chains
atoms_2 = Selection.unfold_entities(str_2, 'C')  # C for chains

# get the mapping from chain,residue id to the atom lists
input_1 = get_atom_list(atoms_1, chains_1)
if len(input_1) == 0:
    print("Are you sure that file " + options.f1 + " has chain(s) " + options.c1 + "? Quitting.")
    quit()
input_2 = get_atom_list(atoms_2, chains_2)
if len(input_2) == 0:
    print("Are you sure that file " + options.f2 + " has chain(s) " + options.c2 + "? Quitting.")
    quit()

# get the full atom lists for neighbor search
all_atoms_1 = get_all_atoms(input_1)
all_atoms_2 = get_all_atoms(input_2)

# run neighbor search on both instances - not optimal but good enough for most imaginable applications.
contacts_1 = get_contacts(input_1, all_atoms_2, "First molecule, residue ", cutoff)
contacts_2 = get_contacts(input_2, all_atoms_1, "Second molecule, residue ", cutoff)

# color the structures according to their new BFactors

contact_map_1 = []
# Color molecule 1
for residue in contacts_1:
    for atom in input_1[residue]:
        atom.set_bfactor(100.0)
        contact_map_1.append(atom)

# Color molecule 2
contact_map_2 = []
for residue in contacts_2:
    for atom in input_2[residue]:
        atom.set_bfactor(100.0)
        contact_map_2.append(atom)

# Get interfacial residues
# run neighbor search on both instances - not optimal but good enough for most imaginable applications.
interface_1 = contacts_1
interface_2 = contacts_2
if (i_cutoff > 0):
    interface_1 = get_contacts(input_1, contact_map_1, "First molecule, interfacial residue ", i_cutoff)
    interface_2 = get_contacts(input_2, contact_map_2, "Second molecule, interfacial residue ", i_cutoff)

# Color molecule 1
for residue in interface_1:
    if residue in contacts_1:
        continue

    for atom in input_1[residue]:
        atom.set_bfactor(50.0)

# Color interface residues on molecule 2
for residue in interface_2:
    if residue in contacts_2:
        continue
    for atom in input_2[residue]:
        atom.set_bfactor(50.0)

# Save the contact-colored structures
save_contacts(str_1, chains_1, cwd + "/" + out_folder + "/molecule_1.pdb")
save_contacts(str_2, chains_2, cwd + "/" + out_folder + "/molecule_2.pdb")

# save the interface residues
save_residues(cwd + "/" + out_folder + "/molecule_1.txt", interface_1, contacts_1)
save_residues(cwd + "/" + out_folder + "/molecule_2.txt", interface_2, contacts_2)

f = open(cwd + "/" + out_folder + "/parameters.txt", 'w')
f.write("The results in this folder were obtained using the following parameters:\n")
f.write("Contact distance: " + str(cutoff) + "A\n")
f.write("Interface distance: " + str(i_cutoff) + "A\n")
f.close()
# Constrain the pdb files to the interface residues only
save_constrained(cwd + "/" + out_folder + "/molecule_1.pdb", cwd + "/" + out_folder + "/molecule_1_constrained.pdb",
                 interface_1)
save_constrained(cwd + "/" + out_folder + "/molecule_2.pdb", cwd + "/" + out_folder + "/molecule_2_constrained.pdb",
                 interface_2)

print("Results saved in " + cwd + "/" + out_folder)
