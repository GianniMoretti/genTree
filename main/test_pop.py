import numpy as np
from genTree.genTree import genTree
from genTree.utils import plot_tree

# Parametri
n_samples = 50
n_features = 4
n_classes = 3

# Dati random di classificazione
X = np.random.rand(n_samples, n_features)
y = np.random.randint(0, n_classes, size=n_samples).astype(np.int32)

# Crea un oggetto genTree
tree = genTree(max_depth=10, min_samples_leaf=2, is_regression=False, pop_size=5)

# Crea popolazione random
tree.create_population(X, y, n_classes)

# Nomi fittizi per le feature e le classi
fake_feature_names = [f"feat_{i}" for i in range(n_features)]
fake_class_names = [f"Classe_{i}" for i in range(n_classes)]

# Visualizza il primo albero della popolazione
first_tree = tree.population[0].root if hasattr(tree.population[0], "root") else tree.population[0]
plot_tree(first_tree, filename="graphs/population_tree", features_name=fake_feature_names, class_names=fake_class_names, is_regression=False)

newtree = tree.split(first_tree, X, y)
plot_tree(newtree, filename="graphs/population_tree_muted", features_name=fake_feature_names, class_names=fake_class_names, is_regression=False)