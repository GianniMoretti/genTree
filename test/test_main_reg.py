import numpy as np
from genTree.genTree import genTree
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt
from genTree.utils import plot_tree as plot_tree_genTree

def generate_dag_data(n_samples=200, n_features=5, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    # Simula un DAG: ogni feature dipende linearmente dalle precedenti
    for j in range(1, n_features):
        X[:, j] += 0.5 * X[:, j-1]
    # y dipende da alcune feature in modo non lineare
    y = 2 * X[:, 0] - 1.5 * X[:, 1] + np.sin(X[:, 2]) + 0.5 * X[:, 3]**2 + np.random.randn(n_samples) * 0.1
    return X, y

if __name__ == "__main__":
    X, y = generate_dag_data(1000)
    y_reg = y.astype(np.float64)
    # Split train/test
    n = X.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(n * 0.8)
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_reg[train_idx], y_reg[test_idx]

    tree = genTree(is_regression=True, max_depth=3, min_samples_leaf=5, expand_prob = 0.6, pop_size=100, n_generations=1000)
    tree.fit(X_train, y_train, alpha=0.25, importance=2, mutation_prob=0.1, best_sel=0.1)
    y_pred = tree.predict(X_test)
    mse = np.mean((y_pred - y_test) ** 2)
    print("Gen tree test MSE:", mse)

    # Salva il grafico dell'albero genTree
    # Nomi fittizi per le feature e le classi
    fake_feature_names = [f"feat_{i}" for i in range(n)]
    plot_tree_genTree(tree.best_tree, filename="graphs/gentree", features_name=fake_feature_names, is_regression=True)

    # Baseline con DecisionTreeRegressor di sklearn
    dtree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=42)
    dtree.fit(X_train, y_train)
    y_pred_dt = dtree.predict(X_test)
    mse_dt = np.mean((y_pred_dt - y_test) ** 2)
    print("DecisionTreeRegressor Test MSE:", mse_dt)

    # Salva il grafico dell'albero sklearn
    plt.figure(figsize=(16, 8))
    plot_tree(dtree, filled=True, feature_names=[f"X{i}" for i in range(X.shape[1])])
    plt.savefig("graphs/sklearn_tree.png")
    plt.close()
    print("Salvato grafico sklearn_tree.png")
