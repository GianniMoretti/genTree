import numpy as np
from genTree import genTree
from genTree.utils import plot_tree

def test_simple_tree():
    # Costruiamo un piccolo dataset fittizio
    X = np.array([[0.1, 1.2],
                    [1.5, 0.3],
                    [0.2, 1.1],
                    [1.4, 0.4]], dtype=np.double)
    y = np.array([0, 1, 0, 1], dtype=np.int32)

    # Inizializziamo un albero con foglia unica
    gt = genTree(max_depth=3, min_samples_leaf=1)
    gt.initialize_random(X, y, n_classes=2)

    # Dovrebbe esistere una root non-NULL
    assert gt.root is not None

    # Calcoliamo fitness (dovrebbe restituire qualcosa senza eccezioni)
    f = gt.compute_fitness(X, y, 0.01)
    assert isinstance(f, float)

    # Predizione su un esempio
    pred = gt.root.predict([0.05, 1.0])
    assert isinstance(pred, float) or isinstance(pred, (int,))

    print("Test superato!")

test_simple_tree()

def test_plot_tree():
    X = np.array([[0.1, 1.2],
                [1.5, 0.3],
                [0.2, 1.1],
                [1.4, 0.4]], dtype=np.double)
    y = np.array([0, 1, 0, 1], dtype=np.int32)
    gt = genTree(max_depth=1, min_samples_leaf=1)
    gt.initialize_random(X, y, n_classes=2)
    # Visualizza l'albero (apre il viewer di graphviz)
    plot_tree(gt.root)
    # Salva l'albero come immagine
    plot_tree(gt.root, filename="graph/test_tree")

test_plot_tree()
