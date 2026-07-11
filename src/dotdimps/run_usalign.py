#!/usr/bin/env python3
"""
Run topological analysis using USalign for initial chain matching and superposition,
followed by internal sequence alignment and gap filling.

USalign must be compiled and placed at:
    <project_root>/USalign/USalign

Usage:
    python run_usalign.py <pdb_reference> <pdb_model>

Example:
    python run_usalign.py data/raw/H1208.pdb data/raw/H1208TS008_1.pdb
"""
import sys
import os
import argparse
import subprocess
import numpy as np

# ── path setup ──────────────────────────────────────────────────────────────
_src = os.path.dirname(__file__)
_root = os.path.abspath(os.path.join(_src, "..", ".."))
sys.path.append(os.path.join(_src, "main_sub-functions"))
sys.path.append(os.path.join(_src, "main_sub-functions", "Structural_AlignmentV2 sub-functions"))

from StructuralAlignmentV2USalign import structural_alignment
from TopCheckV2 import OverlapandSelfintersectParallelV3

USALIGN_EXEC   = os.path.join(_root, "USalign", "USalign")
USALIGN_OUTDIR = os.path.join(_root, "data", "USalign_output_folder")

# ── options ──────────────────────────────────────────────────────────────────
OPTIONS = {
    'MaxLength': 105,
    'dmax': 10,
    'Smoothning': 0,
    'AllowEndContractions': 1,
    'MakeFigures': 0,
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
    'StructureSequenceWeight': 1.5608,
    'SeqenceMisAlignmentPenalty': [7.2200, 2.1660],
    'TrimSeqenceAlignment': 0,
    'SequenceAlignmentExtension': 1,
    'InitialAlignmentExactPairs': 1,
}


def run_usalign(pdb_mobile, pdb_reference, output_prefix="aligned_output"):
    """
    Run USalign to superpose pdb_mobile onto pdb_reference.
    Returns the path to the aligned output PDB.
    """
    if not os.path.isfile(USALIGN_EXEC):
        sys.exit(
            f"USalign binary not found at: {USALIGN_EXEC}\n"
            "Download and compile USalign from https://zhanggroup.org/US-align/ "
            "and place the binary at that path."
        )

    os.makedirs(USALIGN_OUTDIR, exist_ok=True)

    # USalign convention: first arg = mobile (gets transformed), second = fixed (reference)
    # -mm 1 enables multimer alignment mode (recommended for multi-chain structures)
    cmd = [USALIGN_EXEC, pdb_mobile, pdb_reference, "-mm", "1", "-o", output_prefix]
    result = subprocess.run(cmd, cwd=USALIGN_OUTDIR, capture_output=True, text=True)

    if result.returncode != 0:
        print("USalign stderr:\n", result.stderr)
        sys.exit("USalign failed.")

    aligned_pdb = os.path.join(USALIGN_OUTDIR, output_prefix + ".pdb")
    if not os.path.exists(aligned_pdb):
        sys.exit(f"USalign finished but output PDB not found at: {aligned_pdb}")

    print("USalign output summary:")
    for line in result.stdout.splitlines():
        if any(tag in line for tag in ("TM-score", "RMSD", "Aligned length")):
            print(" ", line.strip())
    return aligned_pdb


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdb_reference", help="Reference / target PDB file (fixed)")
    parser.add_argument("pdb_model",     help="Model / prediction PDB file (mobile, aligned by USalign)")
    parser.add_argument("--figures", action="store_true",
                        help="Generate alignment and self-intersection figures")
    args = parser.parse_args()

    pdb_reference = os.path.abspath(args.pdb_reference)
    pdb_model     = os.path.abspath(args.pdb_model)

    for f in (pdb_reference, pdb_model):
        if not os.path.exists(f):
            sys.exit(f"Error: file not found: {f}")

    OPTIONS['FileName1'] = os.path.basename(pdb_reference)
    OPTIONS['FileName2'] = os.path.basename(pdb_model)
    OPTIONS['MakeFigures'] = int(args.figures)

    print(f"Reference : {pdb_reference}")
    print(f"Model     : {pdb_model}")
    print(f"Method    : USalign pre-alignment + internal gap filling\n")

    # Step 1: USalign — superpose model onto reference
    print("Running USalign...")
    aligned_model = run_usalign(pdb_model, pdb_reference)
    print(f"Aligned model saved to: {aligned_model}\n")

    # Step 2: internal sequence alignment and gap filling using the pre-aligned PDB
    try:
        result = structural_alignment(pdb_reference, aligned_model,
                                      makefigure=OPTIONS['MakeFigures'])
    except Exception as e:
        sys.exit(f"Structural alignment failed: {e}")

    (P1, P2, RePar1, RePar2, IsAligned, NresAverage,
     P1Less4, P2Less4, RePar1Less4, RePar2Less4,
     Insert_points_P1, Insert_points_P,
     b_factors1, b_factors2, chain_name1, chain_name2) = result

    P1_tot      = np.concatenate(list(P1.values()),      axis=0)
    P2_tot      = np.concatenate(list(P2.values()),      axis=0)
    P1Less4_tot = np.concatenate(list(P1Less4.values()), axis=0)
    P2Less4_tot = np.concatenate(list(P2Less4.values()), axis=0)

    index1 = index2 = index3 = index4 = 0
    RePar1_tot = []
    RePar2_tot = []
    RePar1Less4_tot = []
    RePar2Less4_tot = []

    for i in list(RePar2.keys()):
        RePar1_tot.extend(RePar1[i] + np.ones(len(RePar1[i])) * index1)
        index1 += RePar1[i][-1] + 1
        RePar2_tot.extend(RePar2[i] + np.ones(len(RePar2[i])) * index2)
        index2 += RePar2[i][-1] + 1
        RePar1Less4_tot.extend(RePar1Less4[i] + np.ones(len(RePar1Less4[i])) * index3)
        index3 += RePar1Less4[i][-1] + 1
        RePar2Less4_tot.extend(RePar2Less4[i] + np.ones(len(RePar2Less4[i])) * index4)
        index4 += RePar2Less4[i][-1] + 1

    IsAligned_tot      = np.ones(len(RePar2_tot))
    IsAlignedLess4_tot = np.ones(len(RePar2Less4_tot))

    False_lines = np.zeros(len(P1))
    start = -1
    for i, chain in zip(range(len(P1Less4)), P1Less4.keys()):
        False_lines[i] = len(P1Less4[chain]) + start
        start = False_lines[i]
    False_lines = False_lines[:-1]

    ud = OverlapandSelfintersectParallelV3(
        P1Less4_tot, P2Less4_tot,
        RePar1Less4_tot, RePar2Less4_tot, IsAlignedLess4_tot,
        0, 0, NresAverage, OPTIONS, False_lines,
        P1, P2, RePar1_tot, RePar2_tot, IsAligned,
        Insert_points_P1, Insert_points_P,
        b_factors1, b_factors2, chain_name1, chain_name2
    )

    print("\nTable of intersections between chains:\n")
    Table_with_axis = ud[2]
    Table_with_axis = np.vstack((chain_name1, Table_with_axis))
    chain_name2.insert(0, "//")
    Table_with_axis = np.hstack((Table_with_axis, np.array(chain_name2).reshape(-1, 1)))
    print(Table_with_axis)


if __name__ == "__main__":
    main()
