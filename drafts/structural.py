from toolbox import mutate_residue
import rosetta
rosetta.init()

ecori_pose = rosetta.pose_from_file('structures/ecori.pdb')
muni_pose = rosetta.pose_from_file('structures/muni.pdb')

ecori_seq = ecori_pose.sequence()
muni_seq = muni_pose.sequence()

res_pose = rosetta.Pose()
res_pose.assign(muni_pose)

for i, aa in enumerate(res_pose.sequence()):
    if not aa == res_1[i]:
        mutate_residue(res_pose, i, res_1[i])

res_pose_dna = rosetta.Pose()
res_pose_dna.assign(res_pose)

chm = rosetta.core.chemical.ChemicalManager.get_instance()
rts = chm.residue_type_set('fa_standard')
nucs = {
    'a': rosetta.core.conformation.ResidueFactory.create_residue(rts.name_map('ADE')),
    'c': rosetta.core.conformation.ResidueFactory.create_residue(rts.name_map('CYT')),
    'g': rosetta.core.conformation.ResidueFactory.create_residue(rts.name_map('GUA')),
    't': rosetta.core.conformation.ResidueFactory.create_residue(rts.name_map('THY'))
}
com = {'a': 't', 'c': 'g', 't': 'a', 'g': 'c'}

res_pose_dna.replace_residue(3, nucs['g'], True)
res_pose_dna.replace_residue(8, nucs['c'], True)
res_pose_dna.replace_residue(13, nucs['g'], True)
res_pose_dna.replace_residue(18, nucs['c'], True)

