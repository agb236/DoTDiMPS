import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "Final sub-function"))


import numpy as np
from StructuralAlignmentV2 import structural_alignment
from TopCheck import OverlapandSelfintersectParallelV3
sys.path.append(os.path.join(os.path.dirname(__file__), "Final sub-function/Structural_AlignmentV2 sub-functions"))
from PDBP_to_seq import one_PDB_to_seq
import timeit


Adam = 1
if Adam == 1:
    pdb_file1 = "/Users/agb/Desktop/6. Semester/Bachelorprojekt/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB/CRUA_hexamer_positive.pdb"
    pdb_file2 = "/Users/agb/Desktop/6. Semester/Bachelorprojekt/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB/CRU1_hexamer_negative.pdb"
    #pdb_file1 = "/Users/agb/Desktop/Bachelorprojekt/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB/T1132TS462_1o.pdb"
    #pdb_file2 = "/Users/agb/Desktop/Bachelorprojekt/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB/T1132TS462_5o.pdb"
    
    #pdb_file1 = "/Users/agb/Desktop/Bachelorprojekt/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB/T1123TS054_2o.pdb"
    #pdb_file1 = "/Users/agb/Desktop/Bachelorprojekt/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB/fixed.pdb"
    #pdb_file2 = "/Users/agb/Desktop/Bachelorprojekt/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB/T1123TS054_1o.pdb"
    #pdb_file1 = "/Users/agb/Desktop/Bachelorprojekt/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB/fold_2024_05_28_10_37_model_2.pdb"
    #pdb_file2 = "/Users/agb/Desktop/Bachelorprojekt/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB/fold_2024_05_28_10_37_model_0.pdb"
    
    
    # Hæmoglobiner:
    #pdb_file1 = "/Users/agb/Desktop/Bachelorprojekt/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB/1A3N.pdb"
    #pdb_file2 = "/Users/agb/Desktop/Bachelorprojekt/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB/fold_1a3n_model_0.pdb"
    
    #pdb_file1 = "/Users/agb/Desktop/Bachelorprojekt/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB/1y8h.pdb"
    #pdb_file2 = "/Users/agb/Desktop/Bachelorprojekt/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB/fold_1y8h_model_0.pdb"
    
    # Ekstra alphafol vs pdb
    #pdb_file1 = "/Users/agb/Desktop/Bachelorprojekt/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB/1ws4.pdb"
    #pdb_file2 = "/Users/agb/Desktop/Bachelorprojekt/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB/fold_1ws4_model_4.pdb"
    
    # Test
    #pdb_file1 = "/Users/agb/Desktop/Bachelorprojekt/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB/T1187o.pdb"
    #pdb_file1 = "/Users/agb/Desktop/Bachelorprojekt/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB/fixed.pdb"
    #pdb_file2 = "/Users/agb/Desktop/Bachelorprojekt/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB/T1187TS098_4o.pdb"
    #pdb_file1 = "/Users/agb/Desktop/Bachelorprojekt/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB/T1187o.pdb"
    #pdb_file2 = "/Users/agb/Desktop/Bachelorprojekt/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB/T1187o/T1187TS098_4o"
else:
    pdb_file1 = "src/dotdimps/Jonas Seje folders/PDB Files/CRUA_hexamer_positive.pdb"
    pdb_file2 = "src/dotdimps/Jonas Seje folders/PDB Files/CRU1_hexamer_negative.pdb"
    

    # Two predicted structures from AlphaFold on dimer protein using T1104 from https://predictioncenter.org/casp15/target.cgi?id=28&view=all
    # pdb_file1 = "C:/Users/Kapta/Documents/Skole/DTU/6.semester/BP/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB/AlphaFold/fold_t1104dimer_model_0.pdb"
    # pdb_file2 = "C:/Users/Kapta/Documents/Skole/DTU/6.semester/BP/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB/AlphaFold/fold_t1104dimer_model_1.pdb"
    
    # Two predicted structures from AlphaFold on hexamer protein using T1104 from https://predictioncenter.org/casp15/target.cgi?id=28&view=all
    # pdb_file1 = "C:/Users/Kapta/Documents/Skole/DTU/6.semester/BP/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB/AlphaFold/fold_t1104hexamer_model_0.pdb"
    # pdb_file2 = "C:/Users/Kapta/Documents/Skole/DTU/6.semester/BP/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB/AlphaFold/fold_t1104hexamer_model_1.pdb"
    # SKAL KØRES MED Best_chain_pairs = [best_perms[3]] I LINJE 122 STRUCTURAL_ALIGNMENTV2.PY

    # Two predicted structures from AlphaFold on pentamer protein using T1114s3 from https://predictioncenter.org/casp15/target.cgi?id=47&view=all
    # pdb_file1 = "C:/Users/Kapta/Documents/Skole/DTU/6.semester/BP/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB/AlphaFold/fold_t1114s3penta_model_0.pdb"
    # pdb_file2 = "C:/Users/Kapta/Documents/Skole/DTU/6.semester/BP/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB/AlphaFold/fold_t1114s3penta_model_4.pdb"

    # Prediction of 1Y8H (AlphaFold) and ground truth from https://www.rcsb.org/structure/1Y8H 
    # pdb_file1 = "C:/Users/Kapta/Documents/Skole/DTU/6.semester/BP/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB/1y8h.pdb"
    # pdb_file2 = "C:/Users/Kapta/Documents/Skole/DTU/6.semester/BP/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB/AlphaFold/1Y8H/fold_1y8h_model_0.pdb"

    # pdb_file2 = "C:/Users/Kapta/Documents/Skole/DTU/6.semester/BP/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB/PDB/8wwu.pdb"


pdb_file1 = "src/dotdimps/Jonas Seje folders/PDB Files/CRUA_hexamer_positive.pdb"
pdb_file2 = "src/dotdimps/Jonas Seje folders/PDB Files/CRU1_hexamer_negative.pdb"


start = timeit.timeit() 
# P1, seq1, s1, tot_seq1, chain_com,b_factors = one_PDB_to_seq(pdb_file2)

P1, P2, RePar1, RePar2, IsAligned, NresAverage, P1Less4, P2Less4, RePar1Less4, RePar2Less4, Insert_points_P1, Insert_points_P, b_factors1, b_factors2 =  structural_alignment(pdb_file1, pdb_file2, makefigure = 1)
# options = {'Smoothning': 0, 'AllowEndContractions': 0, 'MaxLength': 5, 'MakeFigures': 1}
end = timeit.timeit()
print("Time1:", end - start)
start = timeit.timeit() 
options = {
    'MaxLength': 50,
    'dmax': 10,
    'Smoothning': 0,
    'AllowEndContractions': 0,
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
end = timeit.timeit()
print("Time2:", end - start)
OverlapandSelfintersectParallelV3(P1Less4_tot, P2Less4_tot, RePar1Less4_tot, RePar2Less4_tot, IsAlignedLess4_tot, P1org, P2org, NresAverage, options, False_lines, P1, P2, RePar1_tot, RePar2_tot, IsAligned,Insert_points_P1, Insert_points_P, b_factors1, b_factors2)

"""
'calls': Sort by call count.
'cumulative': Sort by cumulative time.
'filename': Sort by the name of the file in which the function was defined.
'line': Sort by the line number in the file where the function was defined.
'module': Sort by the name of the module in which the function was defined.
'name': Sort by function name.
'nfl': Sort by name, file, and line number.
'pcalls': Sort by primitive call count.
'stdname': Sort by standard name.
'time': Sort by internal time.
"""

import cProfile
import pstats
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# def profile_and_print_stats(func, *args, **kwargs):
#     profiler = cProfile.Profile()
#     profiler.enable()
#     func(*args, **kwargs)
#     profiler.disable()
    
#     stats = pstats.Stats(profiler).sort_stats('time')
    
#     script_times = defaultdict(float)
#     for func_name, info in stats.stats.items():
#         filename = func_name[0]
#         # replace 'your_script_names' with the names of your scripts
#         if any(script in filename for script in ['OverlapandSelfintersectParallelV3.py', 'AlignmentMetaData.py', 'NEAMReparametrizationParallel','SelfintersectionTransversal','ScoreSelfIntcWeightedMatchingReparametrizisedParallelTMP','MakeSelfIntcFigureV3' 'Final.py']):
#             script_times[os.path.basename(filename)] += info[2]  # total time
    
#     plt.barh(list(script_times.keys()), list(script_times.values()), color='blue')
#     plt.xlabel('Total Time')
#     plt.ylabel('Script')
#     plt.title('Script Execution Time')
#     plt.show()

# def my_function(P1Less4_tot, P2Less4_tot, RePar1Less4_tot, RePar2Less4_tot, IsAlignedLess4_tot, P1org, P2org, NresAverage, options, False_lines, P1, P2, RePar1_tot, RePar2_tot, IsAligned,Insert_points_P1, Insert_points_P):
#     OverlapandSelfintersectParallelV3(P1Less4_tot, P2Less4_tot, RePar1Less4_tot, RePar2Less4_tot, IsAlignedLess4_tot, P1org, P2org, NresAverage, options, False_lines, P1, P2, RePar1_tot, RePar2_tot, IsAligned,Insert_points_P1, Insert_points_P)

# profile_and_print_stats(my_function,P1Less4_tot, P2Less4_tot, RePar1Less4_tot, RePar2Less4_tot, IsAlignedLess4_tot, P1org, P2org, NresAverage, options, False_lines, P1, P2, RePar1_tot, RePar2_tot, IsAligned,Insert_points_P1, Insert_points_P)





