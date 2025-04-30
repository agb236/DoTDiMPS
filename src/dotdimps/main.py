import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "main_sub-functions"))
from StructuralAlignmentV2USalign import structural_alignment
from TopCheckV2 import OverlapandSelfintersectParallelV3
sys.path.append(os.path.join(os.path.dirname(__file__), "main_sub-functions/Structural_AlignmentV2 sub-functions"))
from PDBP_to_seq import one_PDB_to_seq

# Get the absolute path to the project root
current = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Construct the path to the raw data folder
data_path = os.path.join(current, "data", "raw")
data_path2 = os.path.join(current,"data", "USalign_output_folder")

pdb_file1 = os.path.join(data_path, "CRU1_hexamer_negative.pdb")
pdb_file2 = os.path.join(data_path2, "aligned_output.pdb")
#pdb_file1 = os.path.join(data_path, "H1208TS008_1.pdb")
#pdb_file2 = os.path.join(data_path2, "aligned_output.pdb")
# pdb_file1 = os.path.join(data_path, "CRUA_hexamer_positive.pdb")
# pdb_file2 = os.path.join(data_path, "CRU1_hexamer_negative.pdb")

#pdb_file1 = os.path.join(data_path, "fold_t1104dimer_model_0.pdb")
#pdb_file2 = os.path.join(data_path, "fold_t1104dimer_model_1.pdb")


# options = {'Smoothning': 0, 'AllowEndContractions': 0, 'MaxLength': 5, 'MakeFigures': 1}
options = {
    'MaxLength': 10,
    'dmax': 10,
    'Smoothning': 0,
    'AllowEndContractions': 1,
    'MakeFigures': 1,
    'MakeAlignmentSeedFigure': 0,
    'MakeFiguresInLastItteration': 1,
    'MakeLocalPlotsOfEssensials': 1,
    'SelfIntcFigCutSize': 10,
    'PrintOut': 0,
    'additionalRMSD': 0,
    'alignmentsmoothing': 0,
    'alignmentsmoothingwidth': 3,
    'AdaptiveSubset': 1,
    'MaxNbrAlignmentSeeds': 7,
    'MaxSeedOverlap': 0.5000,
    'MinSeedLength': 40,
    'OverlapWeight': 4,
    'MaxIter': 20,
    'MaxWindowMisalignment': 1,
    'MaxMisAlignment': 0.0150,
    'MinimalAlignmentLength': 30,
    'FileName1': 'CRUA_hexamer_positive.pdb',
    'FileName2': 'CRUA_hexamer_negative.pdb',
    'StructureSequenceWeight': 1.5608,
    'SeqenceMisAlignmentPenalty': [7.2200  ,  2.1660], 
    'TrimSeqenceAlignment': 0,
    'SequenceAlignmentExtension': 1,
    'InitialAlignmentExactPairs': 1
}

P1, P2, RePar1, RePar2, IsAligned, NresAverage, P1Less4, P2Less4, RePar1Less4, RePar2Less4, Insert_points_P1, Insert_points_P, b_factors1, b_factors2, chain_name1, chain_name2 =  structural_alignment(pdb_file1, pdb_file2, makefigure = options['MakeFigures'])

P1org = 0
P2org = 0

P1_tot = np.concatenate(list(P1.values()), axis = 0)
P2_tot = np.concatenate(list(P2.values()), axis = 0)
P1Less4_tot = np.concatenate(list(P1Less4.values()), axis = 0)
P2Less4_tot = np.concatenate(list(P2Less4.values()), axis = 0)


index1 = 0
index2 = 0
index3 = 0
index4 = 0
RePar1_tot = []
RePar2_tot = []
RePar1Less4_tot = []
RePar2Less4_tot = []

for i in list(RePar2.keys()):
    RePar1_tot.extend(RePar1[i]+np.ones(len(RePar1[i]))*index1)
    index1 += RePar1[i][-1]+1
    RePar2_tot.extend(RePar2[i]+np.ones(len(RePar2[i]))*index2)
    index2 += RePar2[i][-1]+1
    RePar1Less4_tot.extend(RePar1Less4[i]+np.ones(len(RePar1Less4[i]))*index3)
    index3 += RePar1Less4[i][-1]+1
    RePar2Less4_tot.extend(RePar2Less4[i]+np.ones(len(RePar2Less4[i]))*index4)
    index4 += RePar2Less4[i][-1]+1

IsAligned_tot = np.ones(len(RePar2_tot))
IsAlignedLess4_tot = np.ones(len(RePar2Less4_tot))
False_lines = np.zeros(len(P1))

start = -1
for i,chain in zip(range(len(P1Less4)), P1Less4.keys()):
    False_lines[i] = len(P1Less4[chain])+start
    start = False_lines[i]

False_lines = False_lines[:-1]
ud = OverlapandSelfintersectParallelV3(P1Less4_tot, P2Less4_tot, RePar1Less4_tot, RePar2Less4_tot, IsAlignedLess4_tot, P1org, P2org, NresAverage, options, False_lines, P1, P2, RePar1_tot, RePar2_tot, IsAligned,Insert_points_P1, Insert_points_P, b_factors1, b_factors2, chain_name1, chain_name2)

print("Table of intersections between chains:\n")
Table_with_axis = ud[2]
Table_with_axis = np.vstack((chain_name1, Table_with_axis))
chain_name2.insert(0, "//")
Table_with_axis = np.hstack((Table_with_axis, np.array(chain_name2).reshape(-1, 1)))


print(Table_with_axis)