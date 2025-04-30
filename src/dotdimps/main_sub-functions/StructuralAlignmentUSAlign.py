import os
import subprocess

# Get the root of the project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",".."))

# Paths
usalign_exec = os.path.join(project_root, "USalign", "USalign")
raw_data_path = os.path.join(project_root, "data", "raw")
processed_data_path = os.path.join(project_root, "data", "USalign_output_folder")

# Input PDBs
pdb_file1 = os.path.join(raw_data_path, "H1208TS008_1.pdb")
pdb_file2 = os.path.join(raw_data_path, "H1208.pdb")

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
