# decisionNode.pxd

cdef class DecisionNode:
    # ——————————————————————————
    # dichiarazione dei campi C-level
    # ——————————————————————————
    cdef public bint is_leaf
    cdef public int feature_index
    cdef public double threshold
    cdef public double prediction
    cdef public int depth
    cdef public int leaf_samples

    # riferimenti ai figli (istanze DecisionNode)
    cdef public DecisionNode left
    cdef public DecisionNode right

    # ——————————————————————————
    # costruttori statici
    # ——————————————————————————
    @staticmethod
    cdef DecisionNode make_leaf(double pred, int samples, int depth)
    @staticmethod
    cdef DecisionNode make_split(int feat_idx, double thresh, int depth, DecisionNode left, DecisionNode right, int samples)

    # ——————————————————————————
    # clonazione (deep copy) e utility
    # ——————————————————————————
    cdef DecisionNode clone(self)
    cdef int _count_leaves(self)
    cdef double _predict_one(self, double[:] x)
