import numpy as np

class BlossomAlgorithm:
    """
    Implementation of Edmonds' Blossom algorithm for finding maximum weight matching in general graphs.
    """
    def __init__(self, edges, maxcardinality=False):
        self.edges = edges
        self.maxcardinality = maxcardinality
        self.nvertex = max(max(i, j) for i, j, _ in edges) + 1
        self.mate = [-1] * self.nvertex
        self.adj_matrix = np.full((self.nvertex, self.nvertex), -np.inf)
        
        for i, j, wt in edges:
            self.adj_matrix[i, j] = self.adj_matrix[j, i] = wt
    
    def find_max_weight_matching(self):
        """Executes Edmonds' Blossom algorithm to find the maximum weight matching."""
        self._initialize()
        self._augment_matching()
        return self.mate
    
    def _initialize(self):
        """Initializes structures required for the algorithm."""
        self.dualvar = [0] * self.nvertex
        self.label = [0] * self.nvertex
        self.bestedge = [-1] * self.nvertex
        self.queue = []
    
    def _augment_matching(self):
        """Main loop of the Blossom algorithm for augmenting paths."""
        for _ in range(self.nvertex):
            self._clear_labels()
            self._find_augmenting_path()
    
    def _clear_labels(self):
        """Resets labels before each augmentation attempt."""
        self.label = [0] * self.nvertex
    
    def _find_augmenting_path(self):
        """Finds an augmenting path and updates the matching accordingly."""
        for i in range(self.nvertex):
            if self.mate[i] == -1:
                self.queue.append(i)
        
        while self.queue:
            v = self.queue.pop(0)
            for u in range(self.nvertex):
                if self.adj_matrix[v, u] > -np.inf and self.mate[u] == -1:
                    self.mate[v] = u
                    self.mate[u] = v
                    return
    

def maxWeightMatching(edges, maxcardinality=False):
    """
    Computes the maximum weight matching in a general graph using Edmonds' Blossom algorithm.
    
    Parameters:
        edges (list of tuples): List of edges represented as (node1, node2, weight).
        maxcardinality (bool): If True, finds a maximum cardinality matching.
    
    Returns:
        list: A list where index i represents the vertex matched with vertex i.
    """
    blossom = BlossomAlgorithm(edges, maxcardinality)
    return blossom.find_max_weight_matching()
