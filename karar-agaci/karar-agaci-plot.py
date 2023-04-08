import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load the iris dataset
iris = load_iris()
X = iris.data[:, 2:]  # Use only petal length and width as features
y = iris.target

# Create and fit a decision tree classifier
tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X, y)

# Plot the decision tree
plt.figure(figsize=(10.8, 10.8))
plot_tree(tree_clf, feature_names=iris.feature_names[2:], class_names=iris.target_names, filled=True)
plt.show()






















