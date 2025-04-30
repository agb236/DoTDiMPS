import os
import numpy as np
import plotly.graph_objects as go
from Bio.PDB import PDBParser, is_aa

def extract_ca_coords(pdb_file):
    """Parse PDB file and extract CA atom coordinates per chain"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)
    coords_per_chain = {}

    for model in structure:
        for chain in model:
            coords = []
            for residue in chain:
                if is_aa(residue):
                    ca = residue["CA"] if "CA" in residue else None
                    if ca:
                        coords.append(ca.get_coord())
            if coords:
                coords_per_chain[chain.id] = np.array(coords)
    return coords_per_chain

def plot_alignment(pdb_file1, aligned_pdb_file):
    coords1 = extract_ca_coords(pdb_file1)
    coords2 = extract_ca_coords(aligned_pdb_file)

    fig = go.Figure()

    # Add reference structure (blue)
    for chain, coords in coords1.items():
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode='lines', name=f"Original {chain}",
            line=dict(width=6, color='blue')
        ))

    # Add aligned structure (red)
    for chain, coords in coords2.items():
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode='lines', name=f"Aligned {chain}",
            line=dict(width=6, color='red')
        ))

    fig.update_layout(
        title="US-align Structural Alignment",
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode='data'
        )
    )
    fig.show()

# === Edit paths below ===
if __name__ == "__main__":
    pdb_file1 = "/Users/agb/Desktop/DoTDiMPS/data/raw/CRU1_hexamer_negative.pdb"
    aligned_output = "/Users/agb/Desktop/DoTDiMPS/data/USalign_output_folder/aligned_output.pdb"

    plot_alignment(pdb_file1, aligned_output)
