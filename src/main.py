import matplotlib.pyplot as plt
from dataset import generate_spiral_dataset, generate_vertical_dataset, initialize

initialize(0)

X, y = generate_vertical_dataset(3, 100)
plt.scatter(X[:, 0], X[:, 1])
plt.show()