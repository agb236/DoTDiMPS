import os
import subprocess
from Bio.PDB import PDBParser
import plotly.graph_objects as go

# Get the root of the project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Paths
usalign_exec = os.path.join(project_root, "USalign", "USalign")
raw_data_path = os.path.join(project_root, "data", "raw")
processed_data_path = os.path.join(project_root, "data", "USalign_output_folder")

# Input PDBs
pdb_file1 = os.path.join(raw_data_path, "CRU1_hexamer_negative.pdb")
pdb_file2 = os.path.join(raw_data_path, "CRUA_hexamer_positive.pdb")

# Ensure processed directory exists
os.makedirs(processed_data_path, exist_ok=True)

def align_proteins(pdb1, pdb2, output_prefix_name, verbose=False):
    command = [usalign_exec, pdb1, pdb2, "-o", output_prefix_name]
    result = subprocess.run(
        command,
        cwd=processed_data_path,
        capture_output=True,
        text=True
    )

    output_pdb = os.path.join(processed_data_path, output_prefix_name + ".pdb")

    if result.returncode != 0:
        print("US-align failed:")
        print(result.stderr)
    else:
        if os.path.exists(output_pdb):
            print(f"Aligned structure saved at: {output_pdb}")
        else:
            print(f"Aligned structure not found at: {output_pdb}")
        
        if verbose:
            print("\nUS-align alignment summary:")
            print(result.stdout)

    return result.stdout, output_pdb

def extract_ca_coordinates(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    atom = residue["CA"]
                    coords.append(atom.coord)

    coords = list(zip(*coords))  # Separate x, y, z
    return coords

def visualize_alignment(reference_pdb, aligned_pdb):
    x1, y1, z1 = extract_ca_coordinates(reference_pdb)
    x2, y2, z2 = extract_ca_coordinates(aligned_pdb)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=x1, y=y1, z=z1,
        mode='markers+lines',
        marker=dict(size=3),
        line=dict(width=2),
        name='Reference'
    ))

    fig.add_trace(go.Scatter3d(
        x=x2, y=y2, z=z2,
        mode='markers+lines',
        marker=dict(size=3),
        line=dict(width=2),
        name='Aligned'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z'
        ),
        title='3D Structural Alignment (C-alpha atoms)',
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()

# Run alignment
stdout, aligned_pdb = align_proteins(pdb_file1, pdb_file2, "aligned_output", verbose=True)

# Visualize alignment
visualize_alignment(pdb_file1, aligned_pdb)

""" import os
import subprocess

# Get the root of the project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",".."))

# Paths
usalign_exec = os.path.join(project_root, "USalign", "USalign")
raw_data_path = os.path.join(project_root, "data", "raw")
processed_data_path = os.path.join(project_root, "data", "USalign_output_folder")

# Input PDBs
pdb_file1 = os.path.join(raw_data_path, "CRU1_hexamer_negative.pdb")
pdb_file2 = os.path.join(raw_data_path, "CRUA_hexamer_positive.pdb")

# Ensure processed directory exists
os.makedirs(processed_data_path, exist_ok=True)

def align_proteins(pdb1, pdb2, output_prefix_name, verbose=False):
    command = [usalign_exec, pdb1, pdb2, "-o", output_prefix_name]
    result = subprocess.run(
        command,
        cwd=processed_data_path,
        capture_output=True,
        text=True
    )

    output_pdb = os.path.join(processed_data_path, output_prefix_name + ".pdb")

    if result.returncode != 0:
        print("US-align failed:")
        print(result.stderr)
    else:
        if os.path.exists(output_pdb):
            print(f"Aligned structure saved at: {output_pdb}")
        else:
            print(f"Aligned structure not found at: {output_pdb}")
        
        # Optionally show US-align's stdout
        if verbose:
            print("\nUS-align alignment summary:")
            print(result.stdout)

    return result.stdout, output_pdb

stdout, aligned_pdb = align_proteins(pdb_file1, pdb_file2, "aligned_output")
 """