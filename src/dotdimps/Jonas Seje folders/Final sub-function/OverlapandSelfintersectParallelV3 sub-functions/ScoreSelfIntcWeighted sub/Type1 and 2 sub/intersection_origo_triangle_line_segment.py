import numpy as np

def intersection_origo_triangle_line_segment(pts, Lstart, Lslut):
    # format long
    uvt = np.linalg.solve(np.column_stack((pts, -Lslut + Lstart)), Lstart)
    ud = (uvt[0] >= 0) & (uvt[1] >= 0) & (uvt[0] + uvt[1] <= 1) & (uvt[2] >= 0) & (uvt[2] <= 1) # hvis det ikke virker - lav til matrix formulation
    return ud

# pts = np.loadtxt("C:/Users/Kapta/Documents/Skole/DTU/6.semester/BP/Python code/Oversæt/Test txt/IntercectionOrigoTriangle_LineSegment/pts_ind.txt")
# Lstart = np.loadtxt("C:/Users/Kapta/Documents/Skole/DTU/6.semester/BP/Python code/Oversæt/Test txt/IntercectionOrigoTriangle_LineSegment/Lstart_ind.txt")
# Lslut = np.loadtxt("C:/Users/Kapta/Documents/Skole/DTU/6.semester/BP/Python code/Oversæt/Test txt/IntercectionOrigoTriangle_LineSegment/Lend_ind.txt")

# intersection_origo_triangle_line_segment(pts, Lstart, Lslut)