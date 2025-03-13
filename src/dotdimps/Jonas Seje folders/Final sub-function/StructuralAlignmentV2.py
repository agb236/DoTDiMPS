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

def structural_alignment(pdb_file1, pdb_file2, makefigure=1):
    """
    Perform structural alignment of two PDB files.
    
    Parameters:
        pdb_file1 (str): Path to the first PDB file.
        pdb_file2 (str): Path to the second PDB file.
        makefigure (int, optional): Flag to generate visualizations (default is 1).
    
    Returns:
        tuple: Aligned protein structures and related metadata.
    """
    def find_missing_numbers(arr, n):
        return [i for i in range(1, n + 1) if i not in arr]
    
    def find_increasing_subarrays(arr):
        result, result2 = [], []
        current_length = 1
        for i in range(1, len(arr)):
            if arr[i] == arr[i - 1] + 1:
                current_length += 1
            else:
                result.extend(range(1, current_length + 1))
                result2.extend([current_length] * current_length)
                current_length = 1
        result.extend(range(1, current_length + 1))
        result2.extend([current_length] * current_length)
        return result, result2

    P1, P2, seq1, seq2, ref_structure, sample_structure, tot_seq1, tot_seq2, chain_com1, chain_com2, b_factors1, b_factors2 = two_PDB_to_seq(pdb_file1, pdb_file2)
    P1_org, P2_org = copy.deepcopy(P1), copy.deepcopy(P2)
    
    chain_name1, chain_name2 = list(seq1.keys()), list(seq2.keys())
    if len(chain_name1) != len(chain_name2):
        raise ValueError("The number of chains in the two structures is not equal")
    
    # Compute distance matrices
    def compute_distance_matrix(chain_com, chain_names):
        size = len(chain_names)
        dist_matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(i+1, size):
                dist_matrix[i, j] = np.linalg.norm(np.array(chain_com[chain_names[i]]) - np.array(chain_com[chain_names[j]]))
                dist_matrix[j, i] = dist_matrix[i, j]
        return dist_matrix
    
    dist_matrix1 = compute_distance_matrix(chain_com1, chain_name1)
    dist_matrix2 = compute_distance_matrix(chain_com2, chain_name2)
    
    # Find best permutation for alignment
    permutations = list(itertools.permutations(chain_name2))
    best_perm = min(permutations, key=lambda perm: Align_3D(
        np.array([chain_com2[chain] for chain in perm]),
        np.array([chain_com1[chain] for chain in chain_name1]))[2])
    
    # Reorder chains
    P2_Reorder = {best_perm[i]: P2[best_perm[i]] for i in range(len(P2))}
    seq2_Reorder = {best_perm[i]: seq2[best_perm[i]] for i in range(len(seq2))}
    
    # Perform sequence alignment
    aligner = Align.PairwiseAligner()
    alignments = {ch1: aligner.align(seq1[ch1], seq2_Reorder[ch2])[0] for ch1, ch2 in zip(chain_name1, chain_name2)}
    
    # Extract aligned residues
    atoms_to_align1, atoms_to_align2 = {}, {}
    for ch1, ch2 in zip(chain_name1, chain_name2):
        aligned = alignments[ch1].aligned
        atoms_to_align1[ch1] = [i for start, end in aligned[0] for i in range(start, end)]
        atoms_to_align2[ch2] = [i for start, end in aligned[1] for i in range(start, end)]
    
    # Convert lists to arrays and center data
    for chain in P1:
        P1[chain] = np.array(P1[chain]) - np.mean(np.concatenate(list(P1.values()), axis=0), axis=0)
        P2_Reorder[chain] = np.array(P2_Reorder[chain]) - np.mean(np.concatenate(list(P2_Reorder.values()), axis=0), axis=0)
    
    # Align 3D structures
    alignment_pts1 = np.vstack([P1[ch][atoms_to_align1[ch]] for ch in chain_name1])
    alignment_pts2 = np.vstack([P2_Reorder[ch][atoms_to_align2[ch]] for ch in chain_name2])
    transformed_pts, R, rmsd = Align_3D(alignment_pts1, alignment_pts2)
    
    # Apply transformation
    P = {ch: transformed_pts[start:start + len(atoms_to_align2[ch]) - 1] for start, ch in enumerate(chain_name1)}
    
    # Insert missing residues
    repar = {ch: np.linspace(0, len(P[ch]) - 1, len(P[ch])).tolist() for ch in chain_name1}
    repar1 = {ch: np.linspace(0, len(P1[ch]) - 1, len(P1[ch])).tolist() for ch in chain_name1}
    
    # Generate visualization if required
    if makefigure:
        fig = go.Figure()
        for ch in P1:
            fig.add_trace(go.Scatter3d(x=P1[ch][:, 0], y=P1[ch][:, 1], z=P1[ch][:, 2], mode='lines', line=dict(width=9, color='blue'), name=ch))
        for ch in P:
            fig.add_trace(go.Scatter3d(x=P[ch][:, 0], y=P[ch][:, 1], z=P[ch][:, 2], mode='lines', line=dict(width=9, color='red'), name="Aligned " + ch))
        fig.update_layout(title_text="Structural Alignment of Protein Structures")
        fig.show()
    
    return P1, P, repar1, repar, {ch: np.ones(len(repar1[ch])) for ch in repar1}, len(np.concatenate(list(P1_org.values())))
