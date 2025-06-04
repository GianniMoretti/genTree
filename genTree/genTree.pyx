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

#TODO: Cambia tutti gli indici dei for per renderli più veloci
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

    ########################################################
    # Initializzaztion
    ########################################################

    cdef object _initialize_random(self, double[:, :] X, int[:] y, int n_classes, np.ndarray[np.int32_t, ndim=1] sample_indices, int depth=0, bint force_split=True):
        """
        Ricorsivamente genera un albero randomico:
        - Split casuale, espansione figli con probabilità self.expand_prob, max_depth.
        - Il primo split viene sempre fatto (force_split=True), se non riesce si riprova.
        - Nodi foglia: valore di voto (classificazione) o media (regressione).
        """
        #Conversione di sample_indices in np.array cosi diventa veloce

        # Se sample_indices è None, usa tutti gli indici
        cdef int n_total_samples = X.shape[0]
        cdef int n_samples = sample_indices.shape[0]
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
                #TODO: Cambia in modo che usi indice dichiarato
                for i in range(n_samples):
                    mean += y[sample_indices[i]]
                mean /= n_samples
                return True, DecisionNode.make_leaf(mean, n_samples, 0, sample_indices)
            else:
                counts[:] = 0
                for i in range(n_samples):
                    counts[y[sample_indices[i]]] += 1
                best_cls = 0
                max_count = counts[0]
                for i in range(1, n_classes):
                    if counts[i] > max_count:
                        max_count = counts[i]
                        best_cls = i
                return True, DecisionNode.make_leaf(best_cls, n_samples, 0, sample_indices)

        # Se è un problema di classificazione e gli elementi sono tutti della stessa classe, ritorna una foglia
        if not self.is_regression and n_samples > 0:
            counts[:] = 0
            for i in range(n_samples):
                counts[y[sample_indices[i]]] += 1
            if np.max(counts) == n_samples:
                best_cls = np.argmax(counts)
                return True, DecisionNode.make_leaf(best_cls, n_samples, 0, sample_indices)

        # Decide se espandere i figli o fermarsi (eccetto primo split)
        if not force_split and np.random.rand() > self.expand_prob:
            if self.is_regression:
                mean = 0.0
                for i in range(n_samples):
                    mean += y[sample_indices[i]]
                mean /= n_samples
                return True, DecisionNode.make_leaf(mean, n_samples, 0, sample_indices)
            else:
                counts[:] = 0
                for i in range(n_samples):
                    counts[y[sample_indices[i]]] += 1
                best_cls = 0
                max_count = counts[0]
                for i in range(1, n_classes):
                    if counts[i] > max_count:
                        max_count = counts[i]
                        best_cls = i
                return True, DecisionNode.make_leaf(best_cls, n_samples, 0, sample_indices)

        # Split casuale, il primo split DEVE riuscire (force_split=True)
        try_count = 0
        cdef double[::1] feature_values
        cdef double prev_val
        cdef int n_candidates
        cdef double[::1] split_candidates = np.empty(n_samples, dtype=np.float64)
        #Cerca uno split valido, se non riesce dopo 10 tentativi, ritorna un valore false e una foglia di default
        while try_count < 10:
            split_feature = np.random.randint(0, n_features)
            # Estrai i valori della feature per i sample_indices
            feature_values_np = np.asarray([X[sample_indices[i], split_feature] for i in range(n_samples)], dtype=np.float64)
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
            # Se non ci sono candidati, non posso splittare
            if n_candidates == 0:
                try_count += 1
                continue
            # Se siamo qui, ho almeno un candidato
            # Scegli uno split casuale tra i candidati
            idx = np.random.randint(0, n_candidates)
            split_value = split_candidates[idx]
            # Calcola maschere left/right e indici
            left_count = 0
            right_count = 0
            for i in range(n_samples):
                if X[sample_indices[i], split_feature] <= split_value:
                    left_count += 1
                else:
                    right_count += 1
            # Se uno dei due figli non ha campioni, non posso splittare
            if left_count == 0 or right_count == 0:
                try_count += 1
                continue

            # Split valido trovato
            break

        if try_count >= 10:
            return False, DecisionNode.make_leaf(0, n_samples, 0, sample_indices)

        # Se siamo qui, abbiamo trovato uno split valido
        # Ricorsione su figli (costruzione indici left/right in Cython) e conteggio
        cdef np.ndarray[np.int32_t, ndim=1] left_idx = np.empty(left_count, dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim=1] right_idx = np.empty(right_count, dtype=np.int32)
        cdef int l = 0      #servono per popolare gli indici
        cdef int r = 0
        for i in range(n_samples):
            if X[sample_indices[i], split_feature] <= split_value:
                left_idx[l] = sample_indices[i]
                l += 1
            else:
                right_idx[r] = sample_indices[i]
                r += 1

        # Chiamata ricorsiva passando solo gli indici
        ok, left_node = self._initialize_random(X, y, n_classes, left_idx, depth + 1, False)
        ok, right_node = self._initialize_random(X, y, n_classes, right_idx, depth + 1, False)

        return True, DecisionNode.make_split(split_feature, split_value, depth, left_node, right_node, n_samples)

    cdef void _create_population(self, double[:, :] X, int[:] y, int n_classes):
        """
        Crea una popolazione di alberi randomici.
        """
        self.population = []
        cdef int i
        cdef int actual_pop_size = self.pop_size
        #TODO: Cambia in modo che ottenga sempre il numero di alberi giusto
        for i in range(actual_pop_size):
            ok, root = self._initialize_random(X, y, n_classes, np.arange(X.shape[0], dtype=np.int32), 0, True)
            if ok:
                self.population.append(root)

    ########################################################
    # Evaluation
    ########################################################

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

    ########################################################
    # Mutation and crossover
    ########################################################
    ## Controlla che quando splitta non splitti un nodo puro o comunque non splitti in due foglie con lo stesso valore di classe
    cdef _split(self, DecisionNode tree, double[:, :] X, int[:] y):
        """
        Mutazione: cerca una foglia randomica scendendo casualmente nell'albero,
        prova a splittarla con una feature e soglia casuale. Se non riesce dopo n_max_fail, fallisce.
        """
        cdef DecisionNode d = tree.clone()  # Clona l'albero per non modificarlo direttamente
        cdef int n_max_fail = 10
        cdef int fail_count = 0
        cdef int n_features = X.shape[1]
        cdef int n_classes = 1  # default, verrà aggiornato sotto se serve
        cdef DecisionNode node
        cdef DecisionNode parent
        cdef DecisionNode left_node, right_node, split_node
        cdef bint left_child
        cdef int depth
        cdef int n_samples
        cdef np.ndarray[np.int32_t, ndim=1] sample_indices
        cdef double[:, :] X_sub
        cdef int[:] y_sub
        cdef int i, j, k
        cdef int split_feature, left_count, right_count, idx
        cdef double split_value, prev_val
        cdef np.ndarray[np.float64_t, ndim=1] feature_values_np
        cdef double[::1] feature_values
        cdef int n_candidates
        cdef double[::1] split_candidates
        cdef int n_total_samples

        while fail_count < n_max_fail:
            # 1. Cerca una foglia randomica scendendo casualmente
            node = d
            parent = None
            left_child = 0
            while not node.is_leaf:
                parent = node
                if np.random.rand() < 0.5:
                    left_child = 1
                    node = node.left
                else:
                    left_child = 0
                    node = node.right

            # 2. Controlla se la foglia è splittabile
            if parent is None or node.leaf_samples <= self.min_samples_leaf or parent.depth + 1 >= self.max_depth:
                fail_count += 1
                continue

            # 3. Prendi gli indici dei campioni della foglia
            sample_indices = node.sample_indices
            n_samples = sample_indices.shape[0]

            #TODO: Non volgio estrarre gli array, voglio lavorare sui x e y originali
            # Ho forse per fare i controlli ha senso estrarli almeno sono meno elementi da ordnare?
            # 4. Estrai X_sub e y_sub usando np.take per compatibilità Cython
            X_sub = np.take(np.asarray(X), sample_indices, axis=0)
            y_sub = np.take(np.asarray(y), sample_indices, axis=0)

            # 5. Scegli una feature casuale e trova split candidates
            split_feature = np.random.randint(0, n_features)
            feature_values_np = np.asarray(X_sub[:, split_feature], dtype=np.float64).copy()
            feature_values_np.sort()
            feature_values = feature_values_np
            n_candidates = 0
            split_candidates = np.empty(n_samples, dtype=np.float64)
            prev_val = feature_values[0]
            for i in range(1, n_samples):
                if feature_values[i] != prev_val:
                    split_candidates[n_candidates] = 0.5 * (feature_values[i] + prev_val)
                    n_candidates += 1
                    prev_val = feature_values[i]

            # Se non ci sono candidati, non posso splittare
            if n_candidates == 0:
                fail_count += 1
                continue

            idx = np.random.randint(0, n_candidates)
            split_value = split_candidates[idx]

            # 6. Calcola left/right
            left_count = 0
            right_count = 0
            for i in range(n_samples):
                if X_sub[i, split_feature] <= split_value:
                    left_count += 1
                else:
                    right_count += 1

            # Se uno dei due figli non ha campioni, non posso splittare
            if left_count == 0 or right_count == 0:
                fail_count += 1
                continue

            #TODO: Non torna gli indici sample indices adesso si riferiscono a X e y
            # 7. Trovare i valori di predizione per i figli
            if self.is_regression:
                left_mean = 0.0
                right_mean = 0.0
                left_indices = []
                right_indices = []
                for i in range(n_samples):
                    if X[sample_indices[i], split_feature] <= split_value:
                        left_mean += y[sample_indices[i]]
                        left_indices.append(sample_indices[i])
                    else:
                        right_mean += y[sample_indices[i]]
                        right_indices.append(sample_indices[i])
                left_mean /= left_count
                right_mean /= right_count
                left_indices_arr = np.array(left_indices, dtype=np.int32)
                right_indices_arr = np.array(right_indices, dtype=np.int32)
                left_node = DecisionNode.make_leaf(left_mean, left_count, 0, left_indices_arr)
                right_node = DecisionNode.make_leaf(right_mean, right_count, 0, right_indices_arr)
            else:
                n_classes = int(np.max(y_sub)) + 1   #TODO: Da cambiare vorrei ussasse tutti i valori?
                left_counts = np.zeros(n_classes, dtype=np.int32)
                right_counts = np.zeros(n_classes, dtype=np.int32)
                left_indices = []
                right_indices = []
                for i in range(n_samples):
                    if X[sample_indices[i], split_feature] <= split_value:
                        left_counts[y[sample_indices[i]]] += 1
                        left_indices.append(sample_indices[i])
                    else:
                        right_counts[y[sample_indices[i]]] += 1
                        right_indices.append(sample_indices[i])
                left_cls = np.argmax(left_counts)
                right_cls = np.argmax(right_counts)
                # Evita split che producono due foglie con la stessa classe
                if left_cls == right_cls:
                    fail_count += 1
                    continue
                left_indices_arr = np.array(left_indices, dtype=np.int32)
                right_indices_arr = np.array(right_indices, dtype=np.int32)
                left_node = DecisionNode.make_leaf(left_cls, left_count, 0, left_indices_arr)
                right_node = DecisionNode.make_leaf(right_cls, right_count, 0, right_indices_arr)

            new_split = DecisionNode.make_split(split_feature, split_value, parent.depth + 1, left_node, right_node, n_samples)

            if left_child:
                parent.left = new_split
            else:
                parent.right = new_split
            return d
        # Se arrivo qui, non ho trovato una foglia splittabile
        return tree

    

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

    ########################################################
    # Fit and predict
    ########################################################

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

    def split(self, tree, X, y):
        """
        Wrapper Python per _split.
        Restituisce una nuova radice mutata (o l'albero originale se la mutazione fallisce).
        """
        return self._split(tree, X, y)
