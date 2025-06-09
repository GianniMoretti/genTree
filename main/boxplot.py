import matplotlib.pyplot as plt
import pandas as pd

# Dati delle tabelle
accuracy_data = {
    "Model": ["evtree", "rpart", "ctree", "genTree"],
    "MeanAccuracy": [0.6521370, 0.6436127, 0.5970072, 0.6295000],
    "CI_Lower": [0.5471126, 0.5149945, 0.4303936, 0.5208000],
    "CI_Upper": [0.7437539, 0.7616579, 0.6956893, 0.7346000]
}

complexity_data = {
    "Model": ["evtree", "rpart", "ctree", "genTree"],
    "MeanComplexity": [46.68399, 39.33260, 38.95699, 43.95000],
    "CI_Lower": [34.74469, 29.37872, 26.82988, 32.20000],
    "CI_Upper": [59.02574, 48.29378, 53.65976, 56.48000]
}

df_acc = pd.DataFrame(accuracy_data)
df_comp = pd.DataFrame(complexity_data)

# Boxplot accuracy (usando CI come whiskers)
fig, ax = plt.subplots()
ax.boxplot(
    [ [row.CI_Lower, row.MeanAccuracy, row.CI_Upper] for _, row in df_acc.iterrows() ],
    labels=df_acc["Model"]
)
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy by Model")
plt.savefig("graphs/boxplot_accuracy.png")
plt.close()

# Boxplot complessità (usando CI come whiskers)
fig, ax = plt.subplots()
ax.boxplot(
    [ [row.CI_Lower, row.MeanComplexity, row.CI_Upper] for _, row in df_comp.iterrows() ],
    labels=df_comp["Model"]
)
ax.set_ylabel("Complexity")
ax.set_title("Complexity by Model")
plt.savefig("graphs/boxplot_complexity.png")
plt.close()

print("Salvati boxplot_accuracy.png e boxplot_complexity.png in graphs/")
