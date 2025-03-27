import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "MakeFigure_sub"))

import numpy as np
import matplotlib.pyplot as plt
import MakeReParTicks as MRPT
import scipy.ndimage
from scipy import sparse
from scipy.interpolate import griddata
import plotly.graph_objects as go

def expand_array(arr):
            return [i for num in arr for i in range(num, num + 5)]

def MakeSelfIntcFigureV3(P, P1, selfintc, overlap, ud_essensials, RePar1, RePar2, myoptions, chain_change, Intersecting_chain_number_i ,Intersecting_chain_number_j, b_factors1, b_factors2):

    RPxtixlables, RPxticks= MRPT.MakeReParTicks(RePar1,8)
    RPytixlables, RPyticks= MRPT.MakeReParTicks(RePar2,8)

    # fig = plt.figure(figsize =(6,6))
    # #Repeat b_factors for each residue in rows
    # background = np.tile(b_factors1, (len(b_factors1), 1))

    # plt.imshow(background.T, cmap='Greys', alpha=0.5, rasterized=True, interpolation ='nearest')

    # plt.imshow(background.T, cmap='Greys', alpha=0.5)
    plt.ylabel('Index in ' + myoptions["FileName2"])
    plt.xlabel('Index in ' + myoptions["FileName1"])
    plt.title('Overlap in Ångström, RMSD = ' + str(np.round(np.sqrt(np.sum((P - P1)**2) / P.shape[0]), 2)) + 'Å')
    plt.xlim(0, overlap.shape[0]+10)
    plt.ylim(0, overlap.shape[1]+10)
    plt.xticks(chain_change, chain_change.astype(int))
    plt.yticks(chain_change, chain_change.astype(int))
    # plt.colorbar()
    # insert rectangles showing which rows has b values less than 50. 
    r = 0
    i = 0
    while i < len(b_factors2)-1:
        if b_factors2[i] < 50:
            r = i
            while b_factors2[r] < 50 and r < len(b_factors2)-1:
                r += 1
            #draw rectangle from row i to row r all the way across
            plt.fill_between([0, overlap.shape[0]+10], i, r, color='orange', alpha=0.5)
            i = r
        else:
            i += 1
    #     print(i,r)

    r = 0
    i = 0
    while i < len(b_factors1)-1:
        if b_factors1[i] < 50:
            r = i
            while b_factors1[r] < 50 and r < len(b_factors1)-1:
                r += 1
            #draw rectangle from column i to column r all the way down
            plt.fill_between([i, r], 0, overlap.shape[1]+10, color='orange', alpha=0.5)
            i = r
        else:
            i += 1

    for c in range(ud_essensials.shape[0]):
        i = ud_essensials[c, 0]
        j = ud_essensials[c, 1]
        # plt.text(i, j, 'e', color='r', fontsize=13, horizontalalignment='center')
        plt.text(j, i, 'e', color='r', fontsize=13, horizontalalignment='center')

    ii, jj = np.where(selfintc)
    for c in range(len(ii)):
        i = ii[c]
        j = jj[c]
        if ~(np.isin(j,ud_essensials[:,1]) and np.isin(i,ud_essensials[:,0])):
            if np.sum(np.sum(abs([i+1, j+1] - ud_essensials), axis= 1)  == 0, axis  = 0) == 0:
                # plt.text(i + 1, j + 1, 'x', color='b', fontsize=11, horizontalalignment='center')
                plt.text(j + 1, i + 1, 'x', color='b', fontsize=11, horizontalalignment='center')

    #I want to create horizontal and vertical lines at the chain change residues
    if chain_change.shape[0] > 0:
        for i in range(chain_change.shape[0]-1):
            plt.axvline(x=chain_change[i], color='black', linestyle='-')
            plt.axhline(y=chain_change[i], color='black', linestyle='-')

    # create list of chian names as strings
    chain_namesX = []
    chain_namesY = []
    for i in range(len(chain_change)-1):
        chain_namesX.append('Chain' + str(i+1))
        chain_namesY.append('Chain' + chr(65 + i))
    # Add second axis to show the chain change residues
    ax2 = plt.twiny()
    ax2.set_xlim(0, overlap.shape[0]+10)
    ax2.set_xticks(chain_change[:-1]+1/2*np.mean(np.diff(chain_change[:-1])))
    ax2.set_xticklabels(chain_namesX)
    # ax2.set_xlabel('Chain change residues')

    ax3 = plt.twinx()
    ax3.set_ylim(0, overlap.shape[1]+10)
    ax3.set_yticks(chain_change[:-1]+1/2*np.mean(np.diff(chain_change[:-1])))
    ax3.set_yticklabels(chain_namesY)
    # ax3.set_ylabel('Chain change residues')

    # make diagonal line
    plt.plot([0, overlap.shape[0]+10], [0, overlap.shape[1]+10], color='black', linestyle='--')
    plt.draw()
    print('continue computation')
    #save as html
    #plt.savefig('C:/Users/Kapta/Documents/Skole/DTU/6.semester/BP/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/EssPlot.png')


    
    # Create traces for each protein chain
    trace1 = go.Scatter3d(
        x=P[:, 0],
        y=P[:, 1],
        z=P[:, 2],
        mode='lines',
        line=dict(color='blue', width=9),
        name='Chain 1',
        legendgroup= 'Chain 1',
    )

    trace2 = go.Scatter3d(
        x=P1[:, 0],
        y=P1[:, 1],
        z=P1[:, 2],
        mode='lines',
        line=dict(color='red', width=9),
        name='Chain 2',
        legendgroup= 'Chain 2'
    )
    traceInterPol = []
    traceInterPol_copy = []
    for i in range(5):
        traceInterPol.append(go.Scatter3d(
        x = (i+1)/(5+1)*P[:,0] + (1-(i+1)/(5+1))*P1[:, 0],
        y = (i+1)/(5+1)*P[:,1] + (1-(i+1)/(5+1))*P1[:, 1],
        z = (i+1)/(5+1)*P[:,2] + (1-(i+1)/(5+1))*P1[:, 2],
        mode='lines',
        showlegend = bool(np.floor(i/4)),
        line=dict(color='grey', width=2),
        opacity=0.5,
        legendgroup = 'Interpolated line',
        name = 'Interpolated lines'
        ))
        traceInterPol_copy.append(go.Scatter3d(
        x = (i+1)/(5+1)*P[:,0] + (1-(i+1)/(5+1))*P1[:, 0],
        y = (i+1)/(5+1)*P[:,1] + (1-(i+1)/(5+1))*P1[:, 1],
        z = (i+1)/(5+1)*P[:,2] + (1-(i+1)/(5+1))*P1[:, 2],
        mode='lines',
        line=dict(color='grey', width=2),
        opacity=0.5,
        legendgroup = 'Interpolated line',
        showlegend = False,
        name = 'Interpolated lines'
        ))
    
    Essential_residues1 = []
    for c in range(ud_essensials.shape[0]):
        n = ud_essensials[c,0]
        essential_interpol_lines = (np.arange(n, n+2, 1)).astype(int)
        # print(essential_interpol_lines)
        for i in range(5):
            Essential_residues1.append(go.Scatter3d(
            x = (i+1)/(5+1)*P[essential_interpol_lines, 0] + (1-(i+1)/(5+1))*P1[essential_interpol_lines, 0],
            y = (i+1)/(5+1)*P[essential_interpol_lines, 1] + (1-(i+1)/(5+1))*P1[essential_interpol_lines, 1],
            z = (i+1)/(5+1)*P[essential_interpol_lines, 2] + (1-(i+1)/(5+1))*P1[essential_interpol_lines, 2],
            mode='lines',
            name = 'Essential residues',
            line=dict(color='yellow', width=9),
            legendgroup = 'Essential residues',
            showlegend = bool(np.floor(i/4))
            ))

    Essential_residues2 = []
    for c in range(ud_essensials.shape[0]):
        n = ud_essensials[c,1]
        essential_interpol_lines = (np.arange(n, n+2, 1)).astype(int)
        # print(essential_interpol_lines)
        for i in range(5):
            Essential_residues2.append(go.Scatter3d(
            x = (i+1)/(5+1)*P[essential_interpol_lines, 0] + (1-(i+1)/(5+1))*P1[essential_interpol_lines, 0],
            y = (i+1)/(5+1)*P[essential_interpol_lines, 1] + (1-(i+1)/(5+1))*P1[essential_interpol_lines, 1],
            z = (i+1)/(5+1)*P[essential_interpol_lines, 2] + (1-(i+1)/(5+1))*P1[essential_interpol_lines, 2],
            mode='lines',
            name = 'Essential residues',
            line=dict(color='yellow', width=9),
            legendgroup = 'Essential residues',
            showlegend = bool(np.floor(i/4))
            ))

    
    # # Create figure
    # fig = go.Figure(data=[trace1, trace2] + traceInterPol+Essential_residues)

    # # Set layout
    # fig.update_layout(
    #     scene=dict(
    #         xaxis=dict(title='X Label'),
    #         yaxis=dict(title='Y Label'),
    #         zaxis=dict(title='Z Label'),
    #         bgcolor='white',
    #         camera=dict(
    #             up=dict(x=0, y=0, z=1),
    #             center=dict(x=0, y=0, z=0),
    #             eye=dict(x=-1.25, y=-1.25, z=1)
    #         )
    #     ),
    # )
    # # Show plot
    # fig.show()

    if ud_essensials.shape[0] > 0:
        trace1_copy = go.Scatter3d(
            x=P[:, 0],
            y=P[:, 1],
            z=P[:, 2],
            mode='lines',
            line=dict(color='blue', width=9),
            name='Chain 1',
            legendgroup= 'Chain 1',
            showlegend= False
        )

        trace2_copy = go.Scatter3d(
            x=P1[:, 0],
            y=P1[:, 1],
            z=P1[:, 2],
            mode='lines',
            line=dict(color='red', width=9),
            name='Chain 2',
            legendgroup= 'Chain 2',
            showlegend= False
        )

        random_indexes = np.random.choice(ud_essensials.shape[0], min(ud_essensials.shape[0], 5), replace=False)
        # random_indexes = np.arange(ud_essensials.shape[0])
        for c in random_indexes:
            up0 = min(abs(chain_change[chain_change > ud_essensials[c,0]] - ud_essensials[c,0]))-1
            down0 = min(abs(chain_change[chain_change < ud_essensials[c,0]] - ud_essensials[c,0]))-1
            up1 = min(abs(chain_change[chain_change > ud_essensials[c,1]] - ud_essensials[c,1]))-1
            down1 = min(abs(chain_change[chain_change < ud_essensials[c,1]] - ud_essensials[c,1]))-1
            index = ((np.arange(ud_essensials[c,0]-min(10, down0),ud_essensials[c,0]+min(10,up0)).astype(int), np.arange(ud_essensials[c,1]-min(10, down1),ud_essensials[c,1]+min(10, up1)).astype(int)))
            # print(index)

            trace1 = go.Scatter3d(
                x=P[:, 0],
                y=P[:, 1],
                z=P[:, 2],
                mode='lines',
                line=dict(color='blue', width=9),
                name='Chain '+str(int(Intersecting_chain_number_i[c]+1)),
                legendgroup= 'Chain 1',
            )

            trace2 = go.Scatter3d(
                x=P1[:, 0],
                y=P1[:, 1],
                z=P1[:, 2],
                mode='lines',
                line=dict(color='darkblue', width=9),
                name='Chain ' + chr(int(Intersecting_chain_number_i[c]+1)+64),
                legendgroup= 'Chain A'
            )

            trace1_copy = go.Scatter3d(
                x=P[:, 0],
                y=P[:, 1],
                z=P[:, 2],
                mode='lines',
                line=dict(color='red', width=9),
                name='Chain ' + str(int(Intersecting_chain_number_j[c]+1)),
                legendgroup= 'Chain 2'
            )

            trace2_copy = go.Scatter3d(
                x=P1[:, 0],
                y=P1[:, 1],
                z=P1[:, 2],
                mode='lines',
                line=dict(color='darkred', width=9),
                name='Chain ' + chr(int(Intersecting_chain_number_j[c]+1)+64),
                legendgroup= 'Chain B'
            )

            traceInterPol = []
            traceInterPol_copy = []
            for i in range(5):
                traceInterPol.append(go.Scatter3d(
                x = (i+1)/(5+1)*P[:,0] + (1-(i+1)/(5+1))*P1[:, 0],
                y = (i+1)/(5+1)*P[:,1] + (1-(i+1)/(5+1))*P1[:, 1],
                z = (i+1)/(5+1)*P[:,2] + (1-(i+1)/(5+1))*P1[:, 2],
                mode='lines',
                showlegend = bool(np.floor(i/4)),
                line=dict(color='grey', width=2),
                opacity=0.5,
                legendgroup = 'Interpolated line',
                name = 'Interpolated lines'
                ))

                traceInterPol_copy.append(go.Scatter3d(
                x = (i+1)/(5+1)*P[:,0] + (1-(i+1)/(5+1))*P1[:, 0],
                y = (i+1)/(5+1)*P[:,1] + (1-(i+1)/(5+1))*P1[:, 1],
                z = (i+1)/(5+1)*P[:,2] + (1-(i+1)/(5+1))*P1[:, 2],
                mode='lines',
                line=dict(color='grey', width=2),
                opacity=0.5,
                legendgroup = 'Interpolated line',
                showlegend = False,
                name = 'Interpolated lines'
                ))
            
            trace1.x = trace1.x[index[0][index[0]<P.shape[0]]]
            trace1.y = trace1.y[index[0][index[0]<P.shape[0]]]
            trace1.z = trace1.z[index[0][index[0]<P.shape[0]]]

            trace1_copy.x = trace1_copy.x[index[1][index[1]<P.shape[0]]]
            trace1_copy.y = trace1_copy.y[index[1][index[1]<P.shape[0]]]
            trace1_copy.z = trace1_copy.z[index[1][index[1]<P.shape[0]]]

            trace2.x = trace2.x[index[0][index[0]<P.shape[0]]]
            trace2.y = trace2.y[index[0][index[0]<P.shape[0]]]
            trace2.z = trace2.z[index[0][index[0]<P.shape[0]]]
            trace2_copy.x = trace2_copy.x[index[1][index[1]<P.shape[0]]]
            trace2_copy.y = trace2_copy.y[index[1][index[1]<P.shape[0]]]
            trace2_copy.z = trace2_copy.z[index[1][index[1]<P.shape[0]]]

            for i in range(len(traceInterPol)):
                traceInterPol[i].x = traceInterPol[i].x[index[0][index[0]<P.shape[0]]]
                traceInterPol[i].y = traceInterPol[i].y[index[0][index[0]<P.shape[0]]]
                traceInterPol[i].z = traceInterPol[i].z[index[0][index[0]<P.shape[0]]]

                traceInterPol_copy[i].x = traceInterPol_copy[i].x[index[1][index[1]<P.shape[0]]]
                traceInterPol_copy[i].y = traceInterPol_copy[i].y[index[1][index[1]<P.shape[0]]]
                traceInterPol_copy[i].z = traceInterPol_copy[i].z[index[1][index[1]<P.shape[0]]]
            
            # Create figure
            fig = go.Figure(data=[trace1, trace2, trace1_copy, trace2_copy] + traceInterPol+traceInterPol_copy +Essential_residues1[c*5:(c+1)*5]+Essential_residues2[c*5:(c+1)*5])

            # Set layout
            fig.update_layout(
                scene=dict(
                    xaxis=dict(title='X'),
                    yaxis=dict(title='Y'),
                    zaxis=dict(title='Z'),
                    bgcolor='white',
                    camera=dict(
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0),
                        eye=dict(x=-1.25, y=-1.25, z=1)
                    )
                ),
            )
            # Show plot
            # fig.write_html("C:/Users/Kapta/Documents/Skole/DTU/6.semester/BP/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/Essential.html")
            fig.show()
            print("")

        Chain1 = 1
        Chain2 = 2

        index1 = np.arange(chain_change[Chain1-1]+1, chain_change[Chain1], 1).astype(int)
        index2 = np.arange(chain_change[Chain2-1]+1, chain_change[Chain2], 1).astype(int)
        # print(index)

        trace1 = go.Scatter3d(
            x=P[:, 0],
            y=P[:, 1],
            z=P[:, 2],
            mode='lines',
            line=dict(color='blue', width=9),
            name='Chain '+str(int(Chain1)),
        ) # Chain 1

        trace2 = go.Scatter3d(
            x=P1[:, 0],
            y=P1[:, 1],
            z=P1[:, 2],
            mode='lines',
            line=dict(color='darkred', width=9),
            name='Chain ' + str(int(Chain2)),
        ) # Chain 2

        trace1_copy = go.Scatter3d(
            x=P[:, 0],
            y=P[:, 1],
            z=P[:, 2],
            mode='lines',
            line=dict(color='red', width=9),
            name='Chain ' + chr(int(Chain2)+64)
        ) # Chain 2

        trace2_copy = go.Scatter3d(
            x=P1[:, 0],
            y=P1[:, 1],
            z=P1[:, 2],
            mode='lines',
            line=dict(color='darkblue', width=9),
            name='Chain ' + chr(int(Chain1)+64),
        ) # Chain 1

        traceInterPol = []
        traceInterPol_copy = []
        for i in range(5):
            traceInterPol.append(go.Scatter3d(
            x = (i+1)/(5+1)*P[:,0] + (1-(i+1)/(5+1))*P1[:, 0],
            y = (i+1)/(5+1)*P[:,1] + (1-(i+1)/(5+1))*P1[:, 1],
            z = (i+1)/(5+1)*P[:,2] + (1-(i+1)/(5+1))*P1[:, 2],
            mode='lines',
            showlegend = bool(np.floor(i/4)),
            line=dict(color='grey', width=2),
            opacity=0.5,
            legendgroup = 'Interpolated line',
            name = 'Interpolated lines'
            ))

            traceInterPol_copy.append(go.Scatter3d(
            x = (i+1)/(5+1)*P[:,0] + (1-(i+1)/(5+1))*P1[:, 0],
            y = (i+1)/(5+1)*P[:,1] + (1-(i+1)/(5+1))*P1[:, 1],
            z = (i+1)/(5+1)*P[:,2] + (1-(i+1)/(5+1))*P1[:, 2],
            mode='lines',
            line=dict(color='grey', width=2),
            opacity=0.5,
            legendgroup = 'Interpolated line',
            showlegend = False,
            name = 'Interpolated lines'
            ))
        
        trace1.x = trace1.x[index1]
        trace1.y = trace1.y[index1]
        trace1.z = trace1.z[index1] # Chain 1

        trace1_copy.x = trace1_copy.x[index2]
        trace1_copy.y = trace1_copy.y[index2]
        trace1_copy.z = trace1_copy.z[index2] # Chain 2

        trace2.x = trace2.x[index2]
        trace2.y = trace2.y[index2]
        trace2.z = trace2.z[index2] # Chain 2

        trace2_copy.x = trace2_copy.x[index1]
        trace2_copy.y = trace2_copy.y[index1]
        trace2_copy.z = trace2_copy.z[index1] # Chain 1

        for i in range(len(traceInterPol)):
            traceInterPol[i].x = traceInterPol[i].x[index1]
            traceInterPol[i].y = traceInterPol[i].y[index1]
            traceInterPol[i].z = traceInterPol[i].z[index1]

            traceInterPol_copy[i].x = traceInterPol_copy[i].x[index2]
            traceInterPol_copy[i].y = traceInterPol_copy[i].y[index2]
            traceInterPol_copy[i].z = traceInterPol_copy[i].z[index2]
        
        

        Essential_residues_in_chains =  expand_array(np.where((((ud_essensials[:,0] <chain_change[Chain1] ) & (chain_change[Chain1-1] < ud_essensials[:,0])) | ((ud_essensials[:,0] < chain_change[Chain2]) & (chain_change[Chain2-1] < ud_essensials[:,0]))) & (((ud_essensials[:,1] <chain_change[Chain1] ) & (chain_change[Chain1-1] < ud_essensials[:,1])) | ((ud_essensials[:,1] < chain_change[Chain2]) & (chain_change[Chain2-1] < ud_essensials[:,0]))))[0].astype(int)*5)
        import operator
        # Create figure
        if len(Essential_residues_in_chains) > 0:
            fig = go.Figure(data=[trace1, trace2, trace1_copy, trace2_copy] + traceInterPol+traceInterPol_copy + 
                            list(operator.itemgetter(*Essential_residues_in_chains)(Essential_residues1))+list(operator.itemgetter(*Essential_residues_in_chains)(Essential_residues2)))

        # Set layout
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z'),
                bgcolor='white',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=-1.25, y=-1.25, z=1)
                )
            ),
        )
        # Show plot
        # fig.write_html("C:/Users/Kapta/Documents/Skole/DTU/6.semester/BP/Detection-of-topological-changes-in-multimer-protein-structures/Multimer/AxB.html")
        
        fig.show()
        plt.show()



# RePar1 = np.loadtxt("C:/Users/Kapta/Documents/Skole/DTU/6.semester/BP/Python code/Oversæt/Test txt/MakeSelfIntcFigureV3/RePar1.txt")
# RePar2 = np.loadtxt("C:/Users/Kapta/Documents/Skole/DTU/6.semester/BP/Python code/Oversæt/Test txt/MakeSelfIntcFigureV3/RePar2.txt")
# P = np.loadtxt("C:/Users/Kapta/Documents/Skole/DTU/6.semester/BP/Python code/Oversæt/Test txt/MakeSelfIntcFigureV3/P.txt")
# P1 = np.loadtxt("C:/Users/Kapta/Documents/Skole/DTU/6.semester/BP/Python code/Oversæt/Test txt/MakeSelfIntcFigureV3/P1.txt")
# selfintc = np.loadtxt("C:/Users/Kapta/Documents/Skole/DTU/6.semester/BP/Python code/Oversæt/Test txt/MakeSelfIntcFigureV3/selfintc.txt")
# overlap = np.loadtxt("C:/Users/Kapta/Documents/Skole/DTU/6.semester/BP/Python code/Oversæt/Test txt/MakeSelfIntcFigureV3/overlap.txt")
# ud_essensials = np.loadtxt("C:/Users/Kapta/Documents/Skole/DTU/6.semester/BP/Python code/Oversæt/Test txt/MakeSelfIntcFigureV3/ud_essentials.txt")
# ud_essensials = ud_essensials.reshape(1,2)

# options_fig = {
#     'MaxLength': 15,
#     'dmax': 10,
#     'Smoothning': 0,
#     'AllowEndContractions': 0,
#     'MakeFigures': 1,
#     'MakeAlignmentSeedFigure': 0,
#     'MakeFiguresInLastItteration': 1,
#     'MakeLocalPlotsOfEssensials': 1,
#     'SelfIntcFigCutSize': 10,
#     'PrintOut': 0,
#     'additionalRMSD': 0,
#     'alignmentsmoothing': 0,
#     'alignmentsmoothingwidth': 3,
#     'AdaptiveSubset': 1,
#     'MaxNbrAlignmentSeeds': 7,
#     'MaxSeedOverlap': 0.5000,
#     'MinSeedLength': 40,
#     'OverlapWeight': 4,
#     'MaxIter': 20,
#     'MaxWindowMisalignment': 1,
#     'MaxMisAlignment': 0.0150,
#     'MinimalAlignmentLength': 30,
#     'FileName1': 'file1.pdb',
#     'FileName2': 'file2.pdb',
#     'StructureSequenceWeight': 1.5608,
#     'SeqenceMisAlignmentPenalty': [7.2200  ,  2.1660], 
#     'TrimSeqenceAlignment': 0,
#     'SequenceAlignmentExtension': 1,
#     'InitialAlignmentExactPairs': 1
# }

# MakeSelfIntcFigureV3(P, P1, selfintc, overlap, ud_essensials, RePar1, RePar2, options_fig)