import pandas as pd
import numpy as np
from genTree.genTree import genTree
from genTree.utils import plot_tree as plot_tree_genTree
import os

# Carica il dataset
df = pd.read_csv(os.path.join("data", "BBBClub.csv"))
#set seed for reproducibility
np.random.seed(56)

# Trasforma le variabili categoriche binarie in 0/1
for col in ["choice", "gender"]:
    if df[col].dtype == object or not np.issubdtype(df[col].dtype, np.number):
        df[col] = df[col].astype("category").cat.codes

# Definisci X e y
X = df.drop(columns=["choice"]).values.astype(np.float64)
y = df["choice"].values.astype(np.float64)

# Usa tutto il dataset per addestrare e valutare
tree = genTree(is_regression=False, max_depth=2, expand_prob = 0.5, min_samples_leaf=10, pop_size=200, n_generations=100)
tree.fit(X, y, alpha=1, importance=1, mutation_prob=0.1, best_sel=0.05)
y_pred = tree.predict(X)
acc = np.mean(y_pred == y)
print("Gen tree accuracy (tutto il dataset):", acc)
print("Gen tree misclassification error (tutto il dataset):", 1 - acc)

# Salva il grafico dell'albero genTree
features_name = [col for col in df.columns if col != "choice"]
class_names = ["0", "1"]
os.makedirs("graphs", exist_ok=True)
plot_tree_genTree(tree.best_tree, filename="graphs/gentree_BBBClub", features_name=features_name, class_names=class_names, is_regression=False)
print("Salvato grafico gentree_BBBClub.png")
