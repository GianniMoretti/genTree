import pandas as pd
import numpy as np
from genTree.genTree import genTree
from genTree.utils import plot_tree as plot_tree_genTree
import os
import matplotlib.pyplot as plt

# Carica il dataset
df = pd.read_csv(os.path.join("data", "glass_identification.csv"))
# set seed for reproducibility
np.random.seed(42)

# Se ci sono colonne categoriche binarie, trasformale in 0/1
# for col in df.columns:
#     if col != "Type" and (df[col].dtype == object or not np.issubdtype(df[col].dtype, np.number)):
#         df[col] = df[col].astype("category").cat.codes

# Definisci X e y
X = df.drop(columns=["Type"]).values.astype(np.float64)
y = df["Type"].values.astype(np.float64) - 1  # Shift labels from 1-7 to 0-6

# Usa tutto il dataset per addestrare e valutare
tree = genTree(is_regression=False, max_depth=5, expand_prob=0.7, min_samples_leaf=10, pop_size=100, n_generations=1000)
tree.fit(X, y, alpha=1, importance=1, mutation_prob=0.1, best_sel=0.05)
y_pred = tree.predict(X)
acc = np.mean(y_pred == y)
print("Gen tree accuracy (tutto il dataset):", acc)
print("Gen tree misclassification error (tutto il dataset):", 1 - acc)

# Calcolo della complessità
try:
    M = tree.count_leaves()
except Exception:
    M = np.nan
N = len(X)
alpha = 1  # già usato nel fit
print("Numero di foglie (M):", M)
print("Numero di campioni (N):", N)
comp = alpha * M * np.log(N) if not np.isnan(M) else np.nan
print("Complessità (alpha * M * log(N)):", comp)

# Salva il grafico dell'albero genTree
features_name = [col for col in df.columns if col != "Type"]
class_names = [str(c) for c in sorted(df["Type"].unique())]
os.makedirs("graph_glass", exist_ok=True)
plot_tree_genTree(tree.best_tree, filename="graph_glass/gentree_glass_identification", features_name=features_name, class_names=class_names, is_regression=False)
print("Salvato grafico gentree_glass_identification.png")

# Parametri bootstrap
n_bootstrap = 100
alpha = 1
acc_list = []
comp_list = []

np.random.seed(42)
for i in range(n_bootstrap):
    print(f"Bootstrap {i+1}/{n_bootstrap}")
    idx = np.random.choice(len(X), size=len(X), replace=True)
    oob_idx = np.setdiff1d(np.arange(len(X)), np.unique(idx))
    if len(oob_idx) == 0:
        continue
    X_train, y_train = X[idx], y[idx]
    X_test, y_test = X[oob_idx], y[oob_idx]
    tree = genTree(is_regression=False, max_depth=5, expand_prob=0.56, min_samples_leaf=10, pop_size=100, n_generations=1000)
    tree.fit(X_train, y_train, alpha=alpha, importance=1, mutation_prob=0.1, best_sel=0.05)
    y_pred = tree.predict(X_test)
    acc = np.mean(y_pred == y_test)
    try:
        M = tree.count_leaves()
    except Exception:
        M = np.nan
    N = len(X_train)
    comp = alpha * M * np.log(N) if not np.isnan(M) else np.nan
    acc_list.append(acc)
    comp_list.append(comp)

acc_arr = np.array(acc_list)
comp_arr = np.array(comp_list)

def summary_stats(arr):
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return np.mean(arr), np.nan, np.nan
    return np.mean(arr), np.quantile(arr, 0.025), np.quantile(arr, 0.975)

acc_mean, acc_lo, acc_hi = summary_stats(acc_arr)
comp_mean, comp_lo, comp_hi = summary_stats(comp_arr)

print(f"Accuracy media: {acc_mean:.4f} (2.5%: {acc_lo:.4f}, 97.5%: {acc_hi:.4f})")
print(f"Complessità media: {comp_mean:.2f} (2.5%: {comp_lo:.2f}, 97.5%: {comp_hi:.2f})")

# Boxplot accuracy
plt.figure(figsize=(8,6))
plt.boxplot(acc_arr[~np.isnan(acc_arr)], vert=True)
plt.ylabel("OOB Accuracy")
plt.title("OOB Accuracy (Bootstrap, n=250)")
plt.savefig("graph_glass/gentree_glass_accuracy_boxplot.png")
plt.close()

# Boxplot complessità
plt.figure(figsize=(8,6))
plt.boxplot(comp_arr[~np.isnan(comp_arr)], vert=True)
plt.ylabel("Complexity (alpha * M * log(N))")
plt.title("Model Complexity (Bootstrap, n=250)")
plt.savefig("graph_glass/gentree_glass_complexity_boxplot.png")
plt.close()
