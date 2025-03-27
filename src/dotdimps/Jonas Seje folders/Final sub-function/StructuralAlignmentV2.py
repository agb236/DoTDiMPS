import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "Structural_AlignmentV2 sub-functions"))

from Bio.PDB import PDBParser
import Bio.PDB
from Bio.SeqUtils import IUPACData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Bio.PDB.Polypeptide import PPBuilder, CaPPBuilder
from Bio import Align
from PDBP_to_seq import two_PDB_to_seq, one_PDB_to_seq
from Align_3D import Align_3D
import plotly.graph_objects as go
import itertools
import copy


def structural_alignment(pdb_file1, pdb_file2, makefigure = 1):
    
    def find_missing_numbers(arr, n):
        # Calculate the sum of integers from 1 to n
        total_sum = n * (n + 1) // 2
        
        # Calculate the sum of elements in the array
        arr_sum = sum(arr)
        
        # Calculate the difference to find the sum of missing numbers
        # missing_sum = total_sum - arr_sum
        
        # Find the missing numbers
        missing_numbers = []
        for i in range(1, n + 1):
            if i not in arr:
                missing_numbers.append(i)
        
        return missing_numbers
    
    def find_increasing_subarrays(arr):
        # Initialize the current length and the result list
        current_length = 1
        result = []
        result2 = []

        # Iterate over the array
        for i in range(1, len(arr)):
            # If the current number is one greater than the previous number, increase the current length
            if arr[i] == arr[i - 1] + 1:
                current_length += 1
            else:
                # Otherwise, add the current length to the result list current_length times, and reset it
                result.extend(np.linspace(1, current_length, current_length, dtype=int))
                result2.extend([current_length]*current_length)
                current_length = 1

        # Don't forget to add the last subarray length
        result.extend(np.linspace(1, current_length, current_length, dtype=int))
        result2.extend([current_length]*current_length)

        return result, result2
    
    # Function to calculate distance matrix for a permutation
    def distance_matrix_for_permutation(perm):
        best_perm = None
        min_RMSD = np.inf
        best_perms = []
        for letter in perm:
            com_array2 = np.zeros((len(chain_com2), 3))
            i = 0
            for chain in letter:
                for j in range(len(chain_com2[chain])):
                    com_array2[i,j] = chain_com2[chain][j]
                i+=1
            transformed_pts, R, RMSD = Align_3D(np.array(com_array2), np.array(com_array))
            if min_RMSD >= RMSD:
                min_RMSD = RMSD
                best_perms.append(letter)
        best_perm = best_perms[-1]
        return best_perm, best_perms

    P1, P2, seq1, seq2, ref_structure, sample_structure, tot_seq1, tot_seq2, chain_com1, chain_com2, b_factors1, b_factors2 = two_PDB_to_seq(pdb_file1, pdb_file2)
    
    P1_org = copy.deepcopy(P1)
    P2_org = copy.deepcopy(P2)


    chain_name1 = list(seq1.keys())
    chain_name2 = list(seq2.keys())

    if len(chain_name1) != len(chain_name2):
        raise ValueError("The number of chains in the two structures is not equal")
    
    distance_matrix1 = np.zeros((len(chain_name1), len(chain_name1)))
    distance_matrix2 = np.zeros((len(chain_name2), len(chain_name2)))
    nr_chains = len(chain_name1)

    for i in range(nr_chains):
        for j in range(nr_chains):
            if chain_name1[i] == chain_name1[j]:
                distance_matrix1[i, j] = 0
            else:
                distance_matrix1[i, j] = np.linalg.norm(np.array(chain_com1[chain_name1[i]]) - np.array(chain_com1[chain_name1[j]]))
                distance_matrix1[j, i] = distance_matrix1[i, j]
            if chain_name2[i] == chain_name2[j]:
                distance_matrix2[i, j] = 0
            else:
                distance_matrix2[i, j] = np.linalg.norm(np.array(chain_com2[chain_name2[i]]) - np.array(chain_com2[chain_name2[j]]))
                distance_matrix2[j, i] = distance_matrix2[i, j]

    permutations = list(itertools.permutations(chain_name2))


    i = 0
    com_array = np.zeros((len(chain_com1), 3))

    for chain in chain_com1.keys():
        # Populating the array with values from lists
        for j in range(len(chain_com1[chain])):
            com_array[i,j] = chain_com1[chain][j]
        i += 1

    best_perm, best_perms = distance_matrix_for_permutation(permutations)

    Best_chain_pairs = [best_perm] #[('Chain_A', 'Chain_B', 'Chain_C', 'Chain_D')]

    #Index for best chain pair
    Best_chain_index = 0

    #Reorder chains in P2 and seq2
    P2_Reorder = {Best_chain_pairs[Best_chain_index][i]: P2[Best_chain_pairs[0][i]] for i in range(len(P2))}
    seq2_Reorder = {Best_chain_pairs[Best_chain_index][i]: seq2[Best_chain_pairs[0][i]] for i in range(len(seq2))}

    chain_name1 = list(seq1.keys())
    chain_name2 = list(seq2_Reorder.keys())

    # Start alignment
    aligner = Align.PairwiseAligner()

    align = {}
    for chain1, chain2 in zip(chain_name1, chain_name2):
        alignments = aligner.align(seq1[chain1], seq2[chain2])
        align[chain1] = alignments[0]
    
    atoms_to_be_aligned1 = {}
    atoms_to_be_aligned2 = {}
    for chain1, chain2 in zip(chain_name1, chain_name2):
        Num_holes = align[chain1].aligned[0].shape[0]
        atoms_to_be_aligned1[chain1] = []
        atoms_to_be_aligned2[chain2] = []
        for i in range(Num_holes-1):
            atoms_to_be_aligned1[chain1].extend(range((align[chain1].aligned[0][i][0]),(align[chain1].aligned[0][i][1])))
            atoms_to_be_aligned2[chain2].extend(range((align[chain1].aligned[1][i][0]),(align[chain1].aligned[1][i][1])))

        atoms_to_be_aligned1[chain1].extend(range((align[chain1].aligned[0][Num_holes-1][0]),(align[chain1].aligned[0][Num_holes-1][1])+1))
        atoms_to_be_aligned2[chain2].extend(range((align[chain1].aligned[1][Num_holes-1][0]),(align[chain1].aligned[1][Num_holes-1][1])+1))

    for chain in P1:
        P1[chain] = P1[chain].tolist()
        P2_Reorder[chain] = P2_Reorder[chain].tolist()

        # Extracting the list of lists from P1
        lists1 = P1[chain]
        lists2 = P2_Reorder[chain]

        # Creating a NumPy array with the same length as lists and 3 columns
        P1_array = np.zeros((len(lists1), 3))
        P2_array = np.zeros((len(lists2), 3))
    
        # Populating the array with values from lists
        for i, sublist in enumerate(lists1):
            P1_array[i] = sublist

        for i, sublist in enumerate(lists2):
            P2_array[i] = sublist

        # Replacing the list of lists with the NumPy array
        P1[chain] = P1_array
        P2_Reorder[chain] = P2_array

    mean1 = np.mean(np.concatenate(list(P1.values()),axis=0),axis=0)
    mean2 = np.mean(np.concatenate(list(P2_Reorder.values()),axis=0),axis=0)
    
    #Center the points
    for chain in P1:
        P1[chain] = P1[chain] - mean1
        P2_Reorder[chain] = P2_Reorder[chain] - mean2

    aligment_points1 = np.zeros((0,3))
    aligment_points2 = np.zeros((0,3))

    for chain1, chain2 in zip(P1, P2_Reorder):
        for i in atoms_to_be_aligned1[chain1]:
            aligment_points1 = np.vstack((aligment_points1, P1[chain1][i-1]))
        for i in atoms_to_be_aligned2[chain2]:
            aligment_points2 = np.vstack((aligment_points2, P2_Reorder[chain2][i-1]))

    aligment_points1 = aligment_points1[1:,:]
    aligment_points2 = aligment_points2[1:,:]

    Transformed_points, R, rmsd = Align_3D(aligment_points1, aligment_points2)

    P = {}
    start = 0
    for chain1, chain2 in zip(P1, P2_Reorder):
        P[chain1] = Transformed_points[start:start+len(atoms_to_be_aligned2[chain1])-1]
        start += len(atoms_to_be_aligned2[chain1])
        
        # Find the difference between the two sets
        atoms_not_aligned = set(range(0,len(P1[chain1]))) - set(atoms_to_be_aligned2[chain1])

        # Convert the set to a list
        atoms_not_aligned = sorted(list(atoms_not_aligned))

        for i,j in enumerate(reversed(atoms_not_aligned)):
            P[chain1] = np.insert(P[chain1], j-(5-i), R@P2_Reorder[chain2][j-1], axis=0)

    for chain in P1:
        P1[chain] = P1[chain].tolist()
        P[chain] = P[chain].tolist()

    repar = {}
    repar1 = {}

    for chain in chain_name1:
        repar[chain] = np.linspace(0,len(P[chain])-1,len(P[chain])).tolist()
        repar1[chain] = np.linspace(0,len(P1[chain])-1,len(P1[chain])).tolist()

    indices_target = {}
    indices_query = {}
    
    for key in P:
        indices_target[key] = [i for i, x in enumerate(align[key][1]) if x == "-"]
        indices_query[key]  = [i for i, x in enumerate(align[key][0]) if x == "-"]

        Factor_hole_target, Index_hole_target  = find_increasing_subarrays(indices_target[key])
        Factor_hole_query, Index_hole_query = find_increasing_subarrays(indices_query[key])

        for i in reversed(range(len(indices_target[key]))):
            index = indices_target[key][i]
            alpha = Factor_hole_target[i]/(Index_hole_target[i]+1)
            new_point = [alpha*P[key][index][0]+(1-alpha)*P[key][index+1][0],
                        alpha*P[key][index][1]+(1-alpha)*P[key][index+1][1],
                        alpha*P[key][index][2]+(1-alpha)*P[key][index+1][2]]
            P[key].insert(index+1,new_point)
            repar[key].insert(index+1-(Factor_hole_target[i]-1),index+alpha-(Factor_hole_target[i]-1))

        for i in reversed(range(len(indices_query[key]))):
            index = indices_query[key][i]
            alpha = Factor_hole_query[i]/(Index_hole_query[i]+1)
            new_point = [alpha*P1[key][index][0]+(1-alpha)*P1[key][index+1][0],
                        alpha*P1[key][index][1]+(1-alpha)*P1[key][index+1][1],
                        alpha*P1[key][index][2]+(1-alpha)*P1[key][index+1][2]]
            P1[key].insert(index+1,new_point)
            repar1[key].insert(index+1-(Factor_hole_query[i]-1),index+alpha-(Factor_hole_query[i]-1))

    L1 = {}
    L2 = {}
    Insert_points_P1 = {}
    Insert_points_P = {}
    PLess4 = copy.deepcopy(P)
    P1Less4 = copy.deepcopy(P1)

    ReParLess4 = copy.deepcopy(repar)
    RePar1Less4 = copy.deepcopy(repar1)
    #print("Length of repar[Chain_A]: ", len(repar["Chain_A"]))
    #print("Length of repar1[Chain_A]: ", len(repar1["Chain_A"]))
    # Insert points in linesegments  > 4
    for chain1, chain2 in zip(P1Less4, PLess4):
        n = len(P1Less4[chain1])
        m =  len(PLess4[chain2])
        P1_tmp = np.array(P1Less4[chain1])
        P_tmp = np.array(PLess4[chain2])
        L1[chain1] = np.sqrt(np.sum((P1_tmp[0:n - 1, :] - P1_tmp[1:n, :]) ** 2, axis=1))
        L2[chain2] = np.sqrt(np.sum((P_tmp[0:m - 1, :] - P_tmp[1:m, :]) ** 2, axis=1))
        Lmax = np.maximum((L1[chain1]), (L2[chain2]))
        Long_lines = np.where(Lmax > 4)
        Insert_points_P1[chain1] = np.zeros((n)).tolist()
        Insert_points_P[chain2] = np.zeros((m)).tolist()
        
        for i in reversed(Long_lines[0]):
            P1Less4[chain1].insert(i+1, ((np.array(P1Less4[chain1])[i,:]+np.array(P1Less4[chain1])[i+1,:])/2).tolist())
            Insert_points_P1[chain1].insert(i+1, 1)
            RePar1Less4[chain1].insert(i+1, (RePar1Less4[chain1][i]+RePar1Less4[chain1][i+1])/2)

            PLess4[chain2].insert(i+1, ((np.array(PLess4[chain2])[i,:]+np.array(PLess4[chain2])[i+1,:])/2).tolist())
            Insert_points_P[chain2].insert(i+1, 1)
            ReParLess4[chain2].insert(i+1, (ReParLess4[chain2][i]+ReParLess4[chain2][i+1])/2)

    #print("Length of repar[Chain_A]: ", len(repar["Chain_A"]))
    #print("Length of repar1[Chain_A]: ", len(repar1["Chain_A"]))



    # Lav repar
    if makefigure == 1:
        # #Plot P1, P2 and P in 3d using plotly
        fig = go.Figure()

        for chain in P1.keys():
            fig.add_trace(go.Scatter3d(x=[i[0] for i in P1[chain]], y=[i[1] for i in P1[chain]], z=[i[2] for i in P1[chain]], mode='lines', line=dict(width=9, color = "blue"), name=chain))

        for chain in P.keys():
            fig.add_trace(go.Scatter3d(x=[i[0] for i in P[chain]], y=[i[1] for i in P[chain]], z=[i[2] for i in P[chain]], mode='lines', line=dict(width=9,color = 'red'), name="Aligned "+chain))

        #add plot title
        fig.update_layout(title_text="Structural alignment of protein structures")
        fig.show()


    # print("RMSD of structual alignment " + str(rmsd))

    is_aligned = {}
    NresAverage = {}

    for chain in repar:
        is_aligned[chain] = np.ones(len(repar1[chain]))
        P1[chain] = np.array(P1[chain])
        P[chain] = np.array(P[chain])

    P1org_tot = np.concatenate(list(P1_org.values()), axis = 0)
    P2org_tot = np.concatenate(list(P2_org.values()), axis = 0)
    NresAverage = (len(P1org_tot)+len(P2org_tot))/2

    return P1, P, repar1, repar, is_aligned, NresAverage, P1Less4, PLess4, RePar1Less4, ReParLess4, Insert_points_P1, Insert_points_P, b_factors1, b_factors2


#pdb_file1 = "/Users/agb/Desktop/Bachelor projekt/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB/CRUA_hexamer_positive.pdb"
#pdb_file2 = "/Users/agb/Desktop/Bachelor projekt/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB/CRU1_hexamer_negative.pdb"

#pdb_file1 = "C:/Users/Kapta/Documents/Skole/DTU/6.semester/BP/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB//CRUA_hexamer_positive.pdb"
#pdb_file2 = "C:/Users/Kapta/Documents/Skole/DTU/6.semester/BP/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB/CRU1_hexamer_negative.pdb"

#P1, P, repar1, repar, is_aligned, NresAverage = structural_alignment(pdb_file1, pdb_file2, makefigure = 1)

