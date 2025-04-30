from Bio.PDB import PDBParser
from Bio import pairwise2
from set_options import set_options
from pass_ca_and_sequence_first_chain import pass_ca_and_sequence_first_chain
from rms_structural_initial_alignment import rms_structural_initial_alignment
from salignment_to_aligned_pairs import salignment_to_aligned_pairs
from fill_alignment_gaps import fill_alignment_gaps
import numpy as np

def make_figure(pdb_path1, pdb_path2):
    """
    Loads two PDB files, extracts alpha-carbons and sequences, performs
    alignment, and computes reparametrizations.

    Parameters:
    pdb_path1 (str): Path to first PDB file
    pdb_path2 (str): Path to second PDB file

    Returns:
    tuple: RePar1, RePar2 (np.ndarray)
    """
    options = set_options()

    # Parse both structures
    parser = PDBParser(QUIET=True)
    structure1 = parser.get_structure("struct1", pdb_path1)
    structure2 = parser.get_structure("struct2", pdb_path2)

    xyz1, seq1 = pass_ca_and_sequence_first_chain(structure1)
    xyz2, seq2 = pass_ca_and_sequence_first_chain(structure2)

    # Perform sequence alignment if not TM-align based
    if not options["TMalignBased"]:
        aln = pairwise2.align.globalxs(seq1, seq2, -1, -0.5, one_alignment_only=True)[0]
        Salign = [aln.seqA, aln.alignments[0][1], aln.seqB]
    else:
        raise NotImplementedError("TM-align parsing not yet implemented in this version.")

    Spairs, Sother_pairs = salignment_to_aligned_pairs(Salign, options)
    SpairsTot = np.hstack((Spairs, Sother_pairs))
    SpairsSorted = SpairsTot[:, np.argsort(SpairsTot[0, :])]

    # Initial RMSD alignment (for visualization or overlap checks)
    P1, P2, *_ = rms_structural_initial_alignment(xyz1, xyz2)

    # Reparametrize curves using the aligned residue pairs
    options["MaximalSequenceAlignmentExtension"] = 0
    RePar1, RePar2 = fill_alignment_gaps(P1, P2, SpairsSorted, options)

    return RePar1, RePar2
