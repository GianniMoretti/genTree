import pandas as pd
import numpy as np
from genTree.genTree import genTree
from genTree.utils import plot_tree as plot_tree_genTree
import os

# Carica il dataset
df = pd.read_csv(os.path.join("data", "chessboard.csv"))
# set seed for reproducibility
np.random.seed(56)

# Trasforma le variabili categoriche binarie in 0/1
# for col in ["class"]:
#     if df[col].dtype == object or not np.issubdtype(df[col].dtype, np.number):
#         df[col] = df[col].astype("category").cat.codes

# Definisci X e y
X = df.drop(columns=["class"]).values.astype(np.float64)
y = df["class"].values.astype(np.float64)

# Split train/test
X_train, X_test = X[:2000], X[2000:4000]
y_train, y_test = y[:2000], y[2000:4000]

# Usa solo il trainset per addestrare
tree = genTree(is_regression=False, max_depth=4, expand_prob=0.5, min_samples_leaf=10, pop_size=100, n_generations=1000)
tree.fit(X_train, y_train, alpha=1, importance=1, mutation_prob=0.1, best_sel=0.05)

# Accuracy train
y_pred_train = tree.predict(X_train)
acc_train = np.mean(y_pred_train == y_train)
print("Train set accuracy:", acc_train)
print("Train set misclassification error:", 1 - acc_train)

# Accuracy test
y_pred_test = tree.predict(X_test)
acc_test = np.mean(y_pred_test == y_test)
print("Test set accuracy:", acc_test)
print("Test set misclassification error:", 1 - acc_test)

# Calcolo della complessità
try:
    M = tree.count_leaves()
except Exception:
    M = np.nan
N = len(X_train)
alpha = 1  # già usato nel fit
comp = alpha * M * np.log(N) if not np.isnan(M) else np.nan
print("Complessità (alpha * M * log(N)):", comp)

# Salva il grafico dell'albero genTree
features_name = [col for col in df.columns if col != "class"]
class_names = [str(c) for c in sorted(df["class"].unique())]
os.makedirs("graphs", exist_ok=True)
plot_tree_genTree(tree.best_tree, filename="graphs/gentree_chessboard", features_name=features_name, class_names=class_names, is_regression=False)
print("Salvato grafico gentree_chessboard.png")
