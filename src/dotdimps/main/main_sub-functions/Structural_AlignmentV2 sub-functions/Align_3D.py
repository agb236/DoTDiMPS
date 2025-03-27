from numpy import array, linalg, linspace, transpose, zeros, mean, ndarray, sign, sqrt, sum
from numpy.linalg import eig, svd, lstsq, det


def Align_3D(P1, P2):
    """
    Aligns two sets of 3D points using the Kabsch algorithm.
    The points in P2 are rotated to minimize the RMSD to the points in P1. P1 and P2 should both
    be centered at the origin.
    :param P1: ndarray
    :param P2: ndarray
    :return: ndarray
    """
    assert isinstance(P1, ndarray) and isinstance(P2, ndarray), "P1 and P2 must be ndarrays"
    assert P1.shape == P2.shape, "P1 and P2 must have the same shape"
    n = P1.shape[0]
    meanp = mean(P2,axis=0)
    meanpt = mean(P1, axis = 0)

    q= P2-meanp
    qt = P1-meanpt

    Qt = transpose(qt)
    Q = transpose(q)


    A = Qt@transpose(Q)
    u, s, vh = svd(A, full_matrices=True)
    d = sign(det(u@vh))

    R = u@array([[1,0,0],[0,1,0],[0,0,d]])@(vh)

    transformed_pts = transpose(R@transpose(q))+meanp

    RMSD = sqrt(1/n * linalg.norm(qt-transformed_pts,"fro")**2) #frobenius norm

    return transformed_pts, R, RMSD