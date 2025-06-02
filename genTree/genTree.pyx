# genTree.pyx

# ——————————————————————————
# IMPORT NECESSARI
# ——————————————————————————
import numpy as np
cimport numpy as np

# “cimport” di DecisionNode dal suo .pxd
from genTree.decisionNode cimport DecisionNode

# ——————————————————————————
# IMPLEMENTAZIONE DI genTree
# ——————————————————————————
cdef class genTree:
    """
    Classe per gestire un albero evolutivo.
    - max_depth, min_samples_leaf: parametri di base
    - population: lista di genTree (popolazione corrente)
    """

    cdef int max_depth
    cdef int min_samples_leaf
    cdef public list population
    cdef DecisionNode best_tree
    cdef DecisionNode root
    cdef bint is_regression
    cdef double expand_prob
    cdef int pop_size
    cdef int n_generations
    cdef double alpha

    # ——————————————————————————
    # Cambiare in modo che gli attributi non siano visibili da fuori della classe
    def __cinit__(self, int max_depth=5, int min_samples_leaf=1, bint is_regression=False, double expand_prob=0.5, int pop_size=100, int n_generations=10, double alpha=0.01):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.population = []
        self.best_tree = None
        self.is_regression = is_regression
        self.expand_prob = expand_prob
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.alpha = alpha

    cdef DecisionNode _initialize_random(self, double[:, :] X, int[:] y, int n_classes, int depth=0, bint force_split=True):
        """
        Ricorsivamente genera un albero randomico:
        - Split casuale, espansione figli con probabilità self.expand_prob, max_depth.
        - Il primo split viene sempre fatto (force_split=True), se non riesce si riprova.
        - Nodi foglia: valore di voto (classificazione) o media (regressione).
        """
        cdef int n_samples = X.shape[0]
        cdef int n_features = X.shape[1]
        cdef int i, split_feature, try_count, idx, left_count, right_count
        cdef double split_value, mean
        cdef DecisionNode left_node, right_node
        cdef np.ndarray[np.int32_t, ndim=1] counts = np.zeros(n_classes, dtype=np.int32)
        cdef int best_cls = 0
        cdef int max_count = 0

        # Condizione di stop: max_depth o pochi campioni
        if depth >= self.max_depth or n_samples <= self.min_samples_leaf:
            if self.is_regression:
                mean = 0.0
                for i in range(n_samples):
                    mean += y[i]
                mean /= n_samples
                return DecisionNode.make_leaf(mean, n_samples, 0)
            else:
                counts[:] = 0
                for i in range(n_samples):
                    counts[y[i]] += 1
                best_cls = 0
                max_count = counts[0]
                for i in range(1, n_classes):
                    if counts[i] > max_count:
                        max_count = counts[i]
                        best_cls = i
                return DecisionNode.make_leaf(best_cls, n_samples, 0)

        # Split casuale, il primo split DEVE riuscire (force_split=True)
        try_count = 0
        cdef double[::1] feature_values
        cdef double prev_val
        cdef int n_candidates
        cdef double[::1] split_candidates = np.empty(n_samples, dtype=np.float64)
        while True:
            split_feature = np.random.randint(0, n_features)
            # Estrai i valori della feature come array numpy (per ordinare)
            # Ottimizzato: usa slicing NumPy invece di un ciclo for
            feature_values_np = np.asarray(X[:, split_feature], dtype=np.float64).copy()
            feature_values_np.sort()
            feature_values = feature_values_np
            # Calcola split candidates (media tra valori adiacenti distinti)
            n_candidates = 0
            prev_val = feature_values[0]
            for i in range(1, n_samples):
                if feature_values[i] != prev_val:
                    split_candidates[n_candidates] = 0.5 * (feature_values[i] + prev_val)
                    n_candidates += 1
                    prev_val = feature_values[i]
            if n_candidates == 0:
                try_count += 1
                if try_count > 10:
                    if self.is_regression:
                        mean = 0.0
                        for i in range(n_samples):
                            mean += y[i]
                        mean /= n_samples
                        return DecisionNode.make_leaf(mean, n_samples, 0)
                    else:
                        counts[:] = 0
                        for i in range(n_samples):
                            counts[y[i]] += 1
                        best_cls = 0
                        max_count = counts[0]
                        for i in range(1, n_classes):
                            if counts[i] > max_count:
                                max_count = counts[i]
                                best_cls = i
                        return DecisionNode.make_leaf(best_cls, n_samples, 0)
                continue
            # Scegli uno split casuale tra i candidati
            idx = np.random.randint(0, n_candidates)
            split_value = split_candidates[idx]
            # Calcola maschere left/right e indici
            left_count = 0
            right_count = 0
            for i in range(n_samples):
                if X[i, split_feature] <= split_value:
                    left_count += 1
                else:
                    right_count += 1
            if left_count == 0 or right_count == 0:
                try_count += 1
                if try_count > 10:
                    if self.is_regression:
                        mean = 0.0
                        for i in range(n_samples):
                            mean += y[i]
                        mean /= n_samples
                        return DecisionNode.make_leaf(mean, n_samples, 0)
                    else:
                        counts[:] = 0
                        for i in range(n_samples):
                            counts[y[i]] += 1
                        best_cls = 0
                        max_count = counts[0]
                        for i in range(1, n_classes):
                            if counts[i] > max_count:
                                max_count = counts[i]
                                best_cls = i
                        return DecisionNode.make_leaf(best_cls, n_samples, 0)
                continue
            break

        # Decide se espandere i figli o fermarsi (eccetto primo split)
        if not force_split and np.random.rand() > self.expand_prob:
            if self.is_regression:
                mean = 0.0
                for i in range(n_samples):
                    mean += y[i]
                mean /= n_samples
                return DecisionNode.make_leaf(mean, n_samples, 0)
            else:
                counts[:] = 0
                for i in range(n_samples):
                    counts[y[i]] += 1
                best_cls = 0
                max_count = counts[0]
                for i in range(1, n_classes):
                    if counts[i] > max_count:
                        max_count = counts[i]
                        best_cls = i
                return DecisionNode.make_leaf(best_cls, n_samples, 0)

        # Ricorsione su figli (costruzione indici left/right in Cython)
        cdef np.ndarray[np.int32_t, ndim=1] left_idx = np.empty(left_count, dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim=1] right_idx = np.empty(right_count, dtype=np.int32)
        cdef int l = 0
        cdef int r = 0
        for i in range(n_samples):
            if X[i, split_feature] <= split_value:
                left_idx[l] = i
                l += 1
            else:
                right_idx[r] = i
                r += 1

        # Slicing X e y per i figli
        cdef int left_n = left_idx.shape[0]
        cdef int right_n = right_idx.shape[0]
        cdef double[:, :] left_X = np.empty((left_n, n_features), dtype=np.float64)
        cdef double[:, :] right_X = np.empty((right_n, n_features), dtype=np.float64)
        cdef int[:] left_y = np.empty(left_n, dtype=np.int32)
        cdef int[:] right_y = np.empty(right_n, dtype=np.int32)
        # TODO: qui si può ottimizzare ulteriormente usando slicing NumPy
        for i in range(left_n):
            for idx in range(n_features):
                left_X[i, idx] = X[left_idx[i], idx]
            left_y[i] = y[left_idx[i]]
        for i in range(right_n):
            for idx in range(n_features):
                right_X[i, idx] = X[right_idx[i], idx]
            right_y[i] = y[right_idx[i]]

        left_node = self._initialize_random(left_X, left_y, n_classes, depth + 1, False)
        right_node = self._initialize_random(right_X, right_y, n_classes, depth + 1, False)

        return DecisionNode.make_split(split_feature, split_value, depth, left_node, right_node, n_samples)

    cdef void _create_population(self, double[:, :] X, int[:] y, int n_classes):
        """
        Crea una popolazione di alberi randomici.
        """
        self.population = []
        cdef int i
        cdef int actual_pop_size = self.pop_size
        for i in range(actual_pop_size):
            root = self._initialize_random(X, y, n_classes, True)
            self.population.append(root)

    cdef double _compute_fitness(self, double[:, :] X, int[:] y, double alpha):
        """
        Fitness = accuracy - alpha * (# foglie)
        """
        cdef int n_samples = X.shape[0]
        cdef int i
        cdef double score = 0.0
        cdef double err
        cdef int correct
        if self.is_regression:
            # MSE negativo come fitness
            err = 0.0
            for i in range(n_samples):
                pred = self.root._predict_one(X[i])
                err += (pred - y[i]) * (pred - y[i])
            score = -err / n_samples
        else:
            correct = 0
            for i in range(n_samples):
                if y[i] == self.root._predict_one(X[i]):
                    correct += 1
            score = correct / n_samples
        cdef int n_leaves = self.root._count_leaves()
        return score - alpha * n_leaves

    cdef genTree _crossover(self, genTree other):
        """
        Esempio minimale di crossover:
        - Clona le due radici (rootA, rootB).
        - Se entrambi non sono foglie, crea child radicando rootA,
        e sostituendo rootA.right con clonedi rootB.left.
        - Altrimenti, child è copia di A.
        """
        cdef genTree child = genTree(self.max_depth, self.min_samples_leaf)
        cdef DecisionNode rootA = self.root.clone()
        cdef DecisionNode rootB = other.root.clone()

        if not rootA.is_leaf and not rootB.is_leaf:
            child.root = rootA
            if rootB.left is not None:
                rootA.right = rootB.left.clone()
        else:
            child.root = rootA

        return child

    cdef DecisionNode _fit(self, double[:, :] X, int[:] y, int n_classes):
        """
        Esegue la ricerca evolutiva per trovare il miglior albero.
        """
        cdef int gen, i
        cdef double best_fitness, fitness
        cdef genTree best_individual

        self._create_population(X, y, n_classes)

        best_fitness = -1e9
        best_individual = None

        for gen in range(self.n_generations):
            fitnesses = []
            for tree in self.population:
                fitness = tree._compute_fitness(X, y, self.alpha)
                fitnesses.append(fitness)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = tree

            # Selezione: prendi i migliori (elitism semplice)
            sorted_pop = [x for _, x in sorted(zip(fitnesses, self.population), key=lambda pair: pair[0], reverse=True)]
            survivors = sorted_pop[:max(2, self.pop_size // 2)]

            # Crossover per nuova generazione
            new_population = survivors[:]
            while len(new_population) < self.pop_size:
                parent1 = np.random.choice(survivors)
                parent2 = np.random.choice(survivors)
                child = parent1._crossover(parent2)
                new_population.append(child)
            self.population = new_population

        self.best_tree = best_individual.root
        return self.best_tree

    cdef np.ndarray[np.float64_t, ndim=1] _predict(self, double[:, :] X):
        """
        Predice i risultati usando il best_tree trovato da fit.
        """
        if self.best_tree is None:
            raise ValueError("Devi chiamare fit prima di predict.")
        cdef int n_samples = X.shape[0]
        cdef np.ndarray[np.float64_t, ndim=1] out = np.empty(n_samples, dtype=np.float64)
        cdef int i
        for i in range(n_samples):
            out[i] = self.best_tree._predict_one(X[i])
        return out

    # --- Python wrappers ---

    def create_population(self, X, y, n_classes):
        """
        Wrapper Python per _create_population.
        """
        self._create_population(X, y, n_classes)

    def fit(self, X, y):
        """
        Wrapper Python per _fit.
        Se n_classes non è fornito e il problema è di classificazione, lo calcola da y come numero di valori unici.
        """
        if not self.is_regression:
            n_classes = len(np.unique(y))  #TODO: Cosi prende il numero ma forse sarebbe meglio che prendesse proprio le classi?
        else:
            n_classes = 1
        return self._fit(X, y, n_classes)

    def predict(self, X):
        """
        Wrapper Python per _predict.
        """
        return self._predict(X)


