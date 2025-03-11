import numpy as np
import matplotlib.pyplot as plt
import MakeReParTicks as MRPT
import plotly.graph_objects as go
from scipy import sparse
from scipy.interpolate import griddata

def expand_array(arr):
    """Expands an array by adding consecutive numbers up to +5 for each element."""
    return [i for num in arr for i in range(num, num + 5)]

def MakeSelfIntcFigureV3(P, P1, selfintc, overlap, ud_essentials, RePar1, RePar2, myoptions, chain_change, 
                          Intersecting_chain_number_i, Intersecting_chain_number_j, b_factors1, b_factors2):
    """
    Generates visualizations for self-intersections in protein structures using both 2D and 3D plots.
    
    Parameters:
        P, P1 (np.ndarray): 3D coordinates of two structures.
        selfintc (np.ndarray): Self-intersection matrix.
        overlap (np.ndarray): Overlap matrix for visualization.
        ud_essentials (np.ndarray): Essential intersection points.
        RePar1, RePar2 (np.ndarray): Reparametrization data.
        myoptions (dict): Plot customization options.
        chain_change (np.ndarray): Chain boundary indices.
        Intersecting_chain_number_i, Intersecting_chain_number_j (np.ndarray): Chain indices for intersections.
        b_factors1, b_factors2 (np.ndarray): B-factors for visualization.
    """
    # Generate tick labels for the reparameterization
    RPxtixlables, RPxticks = MRPT.MakeReParTicks(RePar1, 8)
    RPytixlables, RPyticks = MRPT.MakeReParTicks(RePar2, 8)
    
    # Set up plot labels and title
    plt.ylabel(f'Residue number in {myoptions["FileName2"]}')
    plt.xlabel(f'Residue number in {myoptions["FileName1"]}')
    rmsd = np.round(np.sqrt(np.sum((P - P1)**2) / P.shape[0]), 2)
    plt.title(f'Overlap in Ångström, RMSD={rmsd}Å')
    plt.xlim(0, overlap.shape[0] + 10)
    plt.ylim(0, overlap.shape[1] + 10)
    plt.xticks(chain_change, chain_change.astype(int))
    plt.yticks(chain_change, chain_change.astype(int))
    
    # Highlight B-factor regions below threshold
    for i, b_factors in enumerate([b_factors1, b_factors2]):
        r, j = 0, 0
        while j < len(b_factors) - 1:
            if b_factors[j] < 50:
                r = j
                while b_factors[r] < 50 and r < len(b_factors) - 1:
                    r += 1
                plt.fill_between([0, overlap.shape[0] + 10], j, r, color='orange', alpha=0.5) if i == 0 \
                    else plt.fill_between([j, r], 0, overlap.shape[1] + 10, color='orange', alpha=0.5)
                j = r
            else:
                j += 1
    
    # Plot essential intersections
    for c in range(ud_essentials.shape[0]):
        i, j = ud_essentials[c, 0], ud_essentials[c, 1]
        plt.text(j, i, 'e', color='r', fontsize=13, horizontalalignment='center')
    
    # Mark other intersections
    ii, jj = np.where(selfintc)
    for i, j in zip(ii, jj):
        if not (np.isin(j, ud_essentials[:, 1]) and np.isin(i, ud_essentials[:, 0])):
            plt.text(j + 1, i + 1, 'x', color='b', fontsize=11, horizontalalignment='center')
    
    # Draw vertical and horizontal chain separation lines
    for x in chain_change[:-1]:
        plt.axvline(x=x, color='black', linestyle='-')
        plt.axhline(y=x, color='black', linestyle='-')
    
    # Add chain labels
    chain_namesX = [f'Chain{i+1}' for i in range(len(chain_change) - 1)]
    chain_namesY = [f'Chain{chr(65 + i)}' for i in range(len(chain_change) - 1)]
    
    ax2, ax3 = plt.twiny(), plt.twinx()
    ax2.set_xlim(0, overlap.shape[0] + 10)
    ax2.set_xticks(chain_change[:-1] + 1/2 * np.mean(np.diff(chain_change[:-1])))
    ax2.set_xticklabels(chain_namesX)
    
    ax3.set_ylim(0, overlap.shape[1] + 10)
    ax3.set_yticks(chain_change[:-1] + 1/2 * np.mean(np.diff(chain_change[:-1])))
    ax3.set_yticklabels(chain_namesY)
    
    # Plot diagonal reference line
    plt.plot([0, overlap.shape[0] + 10], [0, overlap.shape[1] + 10], color='black', linestyle='--')
    plt.draw()
    
    # Generate 3D visualization using Plotly
    trace1 = go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode='lines', line=dict(color='blue', width=9), name='Chain 1')
    trace2 = go.Scatter3d(x=P1[:, 0], y=P1[:, 1], z=P1[:, 2], mode='lines', line=dict(color='red', width=9), name='Chain 2')
    
    traces_interpolated = [
        go.Scatter3d(
            x=(i+1)/(5+1)*P[:, 0] + (1-(i+1)/(5+1))*P1[:, 0],
            y=(i+1)/(5+1)*P[:, 1] + (1-(i+1)/(5+1))*P1[:, 1],
            z=(i+1)/(5+1)*P[:, 2] + (1-(i+1)/(5+1))*P1[:, 2],
            mode='lines', line=dict(color='grey', width=2), opacity=0.5, name='Interpolated'
        ) for i in range(5)
    ]
    
    fig = go.Figure(data=[trace1, trace2] + traces_interpolated)
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', bgcolor='white'))
    fig.show()
    plt.show()
