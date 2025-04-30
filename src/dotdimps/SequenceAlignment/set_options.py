import numpy as np

def set_options():
    """
    Returns a dictionary of default options for protein alignment.

    Returns:
    dict: Alignment and optimization parameters
    """
    options = {
        "MaxLength": 15,
        "dmax": 10,
        "Smoothning": 0,
        "AllowEndContractions": 0,
        "SuperpositionMethod": 2,
        "ObjectFunction": 4,
        "MakeFigures": 0,
        "CalculateOverlap": 0,
        "MakeAlignmentSeedFigure": 0,
        "MakeFiguresInLastItteration": 1,
        "MakeLocalPlotsOfEssensials": 1,
        "SelfIntcFigCutSize": 20,
        "PrintOut": 0,
        "additionalRMSD": 0,
        "alignmentsmoothing": 1,
        "alignmentsmoothingwidth": 3,
        "AdaptiveSubset": 1,
        "MaxNbrAlignmentSeeds": 7,
        "MaxSeedOverlap": 0.5,
        "MinSeedLength": 40,
        "OverlapWeight": 4,
        "MaxIter": 20,
        "MaxWindowMisalignment": 1,
        "MaxMisAlignment": 0.015,
        "AlignmnetConvergence": 0,
        "RMSDConv": 0.01,
        "MinimalAlignmentLength": 30,
        "FileName1": "file1.pdb",
        "FileName2": "file2.pdb",
        "StructureSequenceWeight": np.pi / 4,
        "StructureSequenceWeight_k": 1,
        "PartialAlignmentMetrics": 1,
        "SequencePenaltyType": 2,
        "SeqenceMisAlignmentPenalty": 0.5 * 3.8**2 * np.array([1, 0.3]),
        "TrimSeqenceAlignment": 0,
        "SequenceAlignmentExtension": 1,
        "MaximalSequenceAlignmentExtension": 1000,
        "InitialAlignmentExactPairs": 1,
        "InitialAlignmentStructural": 1,
        "TMalignBased": 0
    }

    return options
