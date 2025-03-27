import subprocess

def align_proteins(pdb1, pdb2, output_file):
    command = f"./USalign {pdb1} {pdb2} -o {output_file}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout

# Example usage:
output = align_proteins("protein1.pdb", "protein2.pdb", "aligned_output.pdb")
print(output)
