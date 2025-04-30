import numpy as np
import subprocess
from Bio.PDB import PDBParser
from pass_ca_and_sequence_first_chain import pass_ca_and_sequence_first_chain
from alignment_based_reparametrization import alignment_based_reparametrization
from fill_alignment_gaps import fill_alignment_gaps  # optionally
from rms_structural_initial_alignment import rms_structural_initial_alignment

# Placeholder imports for functions not provided
# from rotate_and_translate_tm_align import rotate_and_translate_tm_align
# from overlap_and_selfintersect import overlap_and_selfintersect
# from reparametrize_aligned_part import reparametrize_aligned_part

def run_tm_align_and_rmsd_one_gap_on_cath(sti, file1, file2, tm_executable_path):
    """
    Runs TM-align and compares it with RMSD-based alignment.

    Parameters:
    sti (str): Folder path containing input files
    file1 (str): Filename of first PDB
    file2 (str): Filename of second PDB
    tm_executable_path (str): Path to TM-align executable

    Returns:
    tuple: udTM, udRMS (overlap and intersection scores from both methods)
    """
    # Run TM-align
    output_file = "TestOut2.txt"
    matrix_file = "matrixLO.txt"
    command = f"{tm_executable_path}TMalignLimitedOutput {sti}{file1} {sti}{file2} -o TMLO.sup -m {matrix_file} -a > {output_file}"
    subprocess.run(command, shell=True)

    with open(output_file, 'r') as f:
        _ = f.readline()
        line1 = f.readline().strip()
        line2 = f.readline().strip()
        line3 = f.readline().strip()

    MA = np.loadtxt(matrix_file)

    # Parse PDB files
    parser = PDBParser(QUIET=True)
    pdb1 = parser.get_structure("pdb1", sti + file1)
    pdb2 = parser.get_structure("pdb2", sti + file2)

    Prexyz1 = pass_ca_and_sequence_first_chain(pdb1)
    Prexyz2 = pass_ca_and_sequence_first_chain(pdb2)

    # Placeholder — you need to implement this function
    # xyz1 = rotate_and_translate_tm_align(Prexyz1, MA)
    xyz1 = Prexyz1  # Temporary fallback
    xyz2 = Prexyz2

    # TM-align based reparametrization
    RePar1, RePar2, IsAligned = alignment_based_reparametrization(line1, line2, line3)

    # Placeholder: apply reparam to align curves (requires interpolation)
    # tmp1 = reparametrize_aligned_part(xyz1, RePar1)
    # tmp2 = reparametrize_aligned_part(xyz2, RePar2)

    # RMSD-based alignment with gap
    P1rms, P2rms, *_ , rmsmin = rms_structural_initial_alignment(Prexyz1, xyz2)

    # Placeholder for full aligned output
    P1finalrms, P2finalrms, RePar1rms, RePar2rms, IsAlignedrms = None, None, None, None, None

    # Placeholder for overlap/self-intersection checks
    # udTM = overlap_and_selfintersect(tmp1, tmp2, RePar1, RePar2, IsAligned, xyz1, xyz2)
    # udRMS = overlap_and_selfintersect(P1finalrms, P2finalrms, RePar1rms, RePar2rms, IsAlignedrms, P1rms, P2rms)

    # For now return empty or fake results
    udTM, udRMS = None, None
    return udTM, udRMS
