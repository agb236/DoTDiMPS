import numpy as np
import PlanarityTransversal as PT

def SelfintersectionTransversal(a0, a1, b0, b1):
    """
    Determines if two moving line segments intersect transversally in 3D space.
    
    Parameters:
        a0, a1 (np.ndarray): 3x2 arrays defining the first line segment at t=0 and t=1.
        b0, b1 (np.ndarray): 3x2 arrays defining the second line segment at t=0 and t=1.
    
    Returns:
        list or int:
            - If an intersection occurs, returns [sign, uv[0], uv[1], s], where:
                * sign indicates the orientation of the intersection.
                * uv[0], uv[1] are the intersection parameters.
                * s is the time parameter at which intersection occurs.
            - Otherwise, returns 0 (no intersection).
    """
    ud = 0  # Default return value (no intersection)
    udplan = PT.PlanarityTransversal(a0, a1, b0, b1)
    slist, transversal = udplan[0], udplan[1]  # t values of planarity and their derivatives
    
    cut = 1e-20  # Tolerance threshold for intersection detection
    
    for i, s in enumerate(x for x in slist if x > 0):
        a = (1 - s) * a0 + s * a1
        b = (1 - s) * b0 + s * b1
        M = np.column_stack((a[:, 1] - a[:, 0], b[:, 0] - b[:, 1]))
        k = b[:, 0] - a[:, 0]
        
        if np.linalg.matrix_rank(M) == 1:
            tmp = np.sum(np.cross(M[:, 0], k) ** 2)  # Distance between parallel lines
            if tmp > cut:
                return ud  # No intersection
            else:
                v = M[:, 0]
                ta, tb = np.dot(v, a), np.dot(v, b)
                intersection_length = abs(ta[0] - tb[1]) + abs(ta[1] - tb[0]) - abs(ta[0] - tb[0]) - abs(ta[1] - tb[1])
                if intersection_length > 4e-14:
                    return [np.sign(transversal[i]), [0.5, 0.5], s]
                return ud
        
        uv, _, _, _ = np.linalg.lstsq(M, k, rcond=None)
        if np.all((0 <= uv) & (uv <= 1)) and np.sum((M @ uv - k) ** 2) < cut:
            return [np.sign(transversal[i]), uv[0], uv[1], s]
    
    return ud
