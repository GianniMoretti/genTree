import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
n_train = 2000
n_test = 2000
n_total = n_train + n_test
np.random.seed(42)

# Generate features
X1 = np.random.uniform(0, 4, n_total)
X2 = np.random.uniform(0, 4, n_total)
X_noise = np.random.uniform(0, 1, (n_total, 6))  # X3-X8

# Assign class: alternate in 4x4 chessboard pattern
# Each square is 1x1 in (X1, X2)
square_x = np.floor(X1).astype(int)
square_y = np.floor(X2).astype(int)
y = ((square_x + square_y) % 2).astype(int)

# Build DataFrame
df = pd.DataFrame({
    'X1': X1,
    'X2': X2,
    'X3': X_noise[:, 0],
    'X4': X_noise[:, 1],
    'X5': X_noise[:, 2],
    'X6': X_noise[:, 3],
    'X7': X_noise[:, 4],
    'X8': X_noise[:, 5],
    'class': y
})

# Save to CSV
df.to_csv('data/chessboard.csv', index=False)

# Plot X1 vs X2 colored by class
plt.figure(figsize=(6, 6))
plt.scatter(df['X1'], df['X2'], c=df['class'], cmap='bwr', alpha=0.5, s=10)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Chessboard Dataset: X1 vs X2')
plt.grid(True)
plt.savefig('graph_chessboard/chessboard_plot.png')
plt.show()
