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

    P1, P2, seq1, seq2, ref_structure, sample_structure, tot_seq1, tot_seq2, chain_com1, chain_com2, b_factors1, b_factors2 = two_PDB_to_seq(pdb_file1, pdb_file2)
    
    P1_org = copy.deepcopy(P1)
    P2_org = copy.deepcopy(P2)


    chain_name1 = list(seq1.keys())
    chain_name2 = list(seq2.keys())

    if len(chain_name1) != len(chain_name2):
        raise ValueError("The number of chains in the two structures is not equal")
    
    #Reorder chains in P2 and seq2
    P2_Reorder = P2
    seq2_Reorder = seq2

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

        atoms_to_be_aligned1[chain1].extend(range((align[chain1].aligned[0][Num_holes-1][0]),(align[chain1].aligned[0][Num_holes-1][1])))
        atoms_to_be_aligned2[chain2].extend(range((align[chain1].aligned[1][Num_holes-1][0]),(align[chain1].aligned[1][Num_holes-1][1])))

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

    # Collect aligned residues using correct 0-based indexing
    aligment_points1 = np.vstack([P1[c1][atoms_to_be_aligned1[c1]] for c1 in chain_name1])
    aligment_points2 = np.vstack([P2_Reorder[c2][atoms_to_be_aligned2[c2]] for c2 in chain_name2])

    Transformed_points, R, rmsd = Align_3D(aligment_points1, aligment_points2)

    # Pivot point used by Align_3D's internal centering — needed to apply the same
    # Kabsch transform consistently to non-aligned residues
    mean_aln2 = np.mean(aligment_points2, axis=0)

    # Apply the full Kabsch transformation to every P2 residue (aligned and non-aligned)
    P = {}
    for chain1, chain2 in zip(P1, P2_Reorder):
        n2 = len(P2_Reorder[chain2])
        P[chain1] = np.zeros((n2, 3))
        for j in range(n2):
            P[chain1][j] = R @ (P2_Reorder[chain2][j] - mean_aln2) + mean_aln2

    for chain in P1:
        P1[chain] = P1[chain].tolist()
        P[chain] = P[chain].tolist()
    
    repar = {}
    repar1 = {}

    for chain in chain_name1:
        repar[chain] = np.linspace(0,len(P[chain])-1,len(P[chain])).tolist()
        repar1[chain] = np.linspace(0,len(P1[chain])-1,len(P1[chain])).tolist()

    def fill_coord_gaps(coord_list, repar_list, gap_aln_positions, aln_str_gapped):
        """
        Insert interpolated virtual residues into coord_list and repar_list wherever
        aln_str_gapped has a '-' (meaning the other chain has a residue here but this
        one does not).  gap_aln_positions are the alignment-string column indices of
        those '-' characters.

        The previous implementation used alignment-string positions directly as array
        indices, which crashes when sequences differ in length (alignment string longer
        than residue array).  This version converts each gap column to the correct
        0-based residue index before accessing the coordinate arrays.
        """
        if not gap_aln_positions:
            return coord_list, repar_list

        result_c = list(coord_list)
        result_r = list(repar_list)
        offset = 0  # how many virtual points have been inserted so far

        i = 0
        while i < len(gap_aln_positions):
            # Collect a run of consecutive alignment-string gap positions
            j = i
            while j + 1 < len(gap_aln_positions) and gap_aln_positions[j+1] == gap_aln_positions[j] + 1:
                j += 1
            N = j - i + 1  # number of gaps in this run

            # Convert the first alignment-string position of the run to the residue
            # index in the array being filled: count non-gap chars before it.
            r = sum(1 for c in aln_str_gapped[:gap_aln_positions[i]] if c != '-')
            pos = r + offset  # adjusted position after earlier insertions

            n = len(result_c)
            if 0 < pos < n:
                pt0 = np.array(result_c[pos - 1])
                pt1 = np.array(result_c[pos])
                rp0, rp1 = result_r[pos - 1], result_r[pos]
            elif pos == 0:
                pt0 = pt1 = np.array(result_c[0])
                rp0 = rp1 = result_r[0]
            else:
                pt0 = pt1 = np.array(result_c[-1])
                rp0 = rp1 = result_r[-1]

            # Insert N virtual points in reversed order so the final array is in
            # ascending order (pt0 → interpolated points → pt1).
            for k in range(N, 0, -1):
                alpha = k / (N + 1)
                result_c.insert(pos, ((1 - alpha) * pt0 + alpha * pt1).tolist())
                result_r.insert(pos, rp0 + alpha * (rp1 - rp0))

            offset += N
            i = j + 1

        return result_c, result_r

    for key in P:
        gaps_in_P  = [i for i, x in enumerate(align[key][1]) if x == "-"]
        gaps_in_P1 = [i for i, x in enumerate(align[key][0]) if x == "-"]

        P[key],  repar[key]  = fill_coord_gaps(P[key],  repar[key],  gaps_in_P,  align[key][1])
        P1[key], repar1[key] = fill_coord_gaps(P1[key], repar1[key], gaps_in_P1, align[key][0])

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

        for chain in P2.keys():
            fig.add_trace(go.Scatter3d(x=[i[0] for i in P2[chain]], y=[i[1] for i in P2[chain]], z=[i[2] for i in P2[chain]], mode='lines', line=dict(width=9,color = 'red'), name="Aligned "+chain))

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

    return P1, P, repar1, repar, is_aligned, NresAverage, P1Less4, PLess4, RePar1Less4, ReParLess4, Insert_points_P1, Insert_points_P, b_factors1, b_factors2, chain_name1, chain_name2


#pdb_file1 = "/Users/agb/Desktop/DoTDiMPS/data/raw/CRUA_hexamer_positive.pdb"
#pdb_file2 = "/Users/agb/Desktop/DoTDiMPS/data/USalign_output_folder/aligned_output.pdb"

#pdb_file1 = "C:/Users/Kapta/Documents/Skole/DTU/6.semester/BP/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB//CRUA_hexamer_positive.pdb"
#pdb_file2 = "C:/Users/Kapta/Documents/Skole/DTU/6.semester/BP/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/examples/Multimer PDB/CRU1_hexamer_negative.pdb"

#results = structural_alignment(pdb_file1, pdb_file2, makefigure=1)
