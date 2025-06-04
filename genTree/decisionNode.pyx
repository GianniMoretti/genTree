# decisionNode.pyx

# ——————————————————————————
# IMPORT
# ——————————————————————————
import numpy as np
cimport numpy as np

# (Non servono più import da cpython.mem perché non usiamo più malloc/free)

# ——————————————————————————
# IMPLEMENTAZIONE DI DecisionNode
# ——————————————————————————
cdef class DecisionNode:
    """
    Implementazione di DecisionNode (tutti i campi sono già dichiarati nel .pxd).
    """

    def __cinit__(self):
        # Inizializzo valori di default
        self.is_leaf = True
        self.feature_index = -1
        self.threshold = 0.0
        self.prediction = 0.0
        self.depth = 0
        self.leaf_samples = 0
        self.left = None
        self.right = None
        self.sample_indices = np.empty(0, dtype=np.int32)

    # def __dealloc__(self):
    #     # Il GC di Python dealloca ricorsivamente left/right
    #     self.left = None
    #     self.right = None

    @staticmethod
    cdef DecisionNode make_leaf(double pred, int samples, int depth, object sample_indices):
        """
        Restituisce un nuovo nodo foglia con:
        - prediction = pred
        - leaf_samples = samples
        - depth = depth
        - sample_indices = np.ndarray[int32] (indici degli esempi)
        """
        cdef DecisionNode node = DecisionNode()
        node.is_leaf = True
        node.prediction = pred
        node.leaf_samples = samples
        node.depth = depth
        node.sample_indices = sample_indices
        return node

    @staticmethod
    cdef DecisionNode make_split(int feat_idx, double thresh, int depth, DecisionNode left, DecisionNode right, int samples):
        """
        Restituisce un nuovo nodo split (interno) con:
        - feature_index = feat_idx
        - threshold = thresh
        - depth = depth
        - sample_indices = array vuoto
        """
        cdef DecisionNode node = DecisionNode()
        node.is_leaf = False
        node.feature_index = feat_idx
        node.threshold = thresh
        node.depth = depth
        node.left = left
        node.right = right
        node.leaf_samples = samples
        node.sample_indices = np.empty(0, dtype=np.int32)
        return node

    cdef DecisionNode clone(self):
        """
        Copia profonda (deep copy) di questo nodo e del suo sottoalbero.
        """
        cdef DecisionNode new_node
        import numpy as np
        if self.is_leaf:
            # Copia anche sample_indices (deep copy)
            sample_indices_copy = np.copy(self.sample_indices)
            new_node = DecisionNode.make_leaf(self.prediction, self.leaf_samples, 0, sample_indices_copy)
        else:
            new_node = DecisionNode.make_split(
                self.feature_index, self.threshold, self.depth,
                self.left.clone(), self.right.clone(), self.leaf_samples
            )
        return new_node

    cdef int _count_leaves(self):
        """
        Conta e restituisce il numero di foglie nel sottoalbero.
        """
        if self.is_leaf:
            return 1
        cdef int cnt = 0
        if self.left is not None:
            cnt += self.left._count_leaves()
        if self.right is not None:
            cnt += self.right._count_leaves()
        return cnt

    cdef double _predict_one(self, double[:] x):
        """
        Ricorsivamente, dati i valori delle feature in x, scende lungo l'albero.
        Se è foglia, ritorna prediction; altrimenti usa feature_index e threshold.
        """
        if self.is_leaf:
            return self.prediction
        if x[self.feature_index] <= self.threshold:
            return self.left._predict_one(x)
        else:
            return self.right._predict_one(x)

    def predict(self, x):
        """
        Wrapper Python-friendly per _predict_one:
        - Se x è già un np.ndarray[float64, ndim=1], uso direttamente memoryview.
        - Altrimenti converto con np.array(x, dtype=np.float64).
        - Restituisco un float.
        """
        cdef double[::1] arr
        cdef np.ndarray[np.float64_t, ndim=1] tmp_arr

        # Se x è array numpy 1D di float64
        if isinstance(x, np.ndarray) and x.dtype == np.float64 and x.ndim == 1:
            tmp_arr = x
        else:
            tmp_arr = np.array(x, dtype=np.float64)
        arr = tmp_arr  # Assegna direttamente senza cast esplicito
        return self._predict_one(arr)
