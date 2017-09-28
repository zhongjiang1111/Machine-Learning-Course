# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 22:05:14 2017

@author: Ruan Shuhua
"""

# import datasets and numpy for science computing
from sklearn import datasets
import numpy as np

# loda iris datasets
iris = datasets.load_iris()

# X is dataset, iris'Petal.Length and Petal.Width
# y is label to indicate different categories
X = iris.data[:, [2, 3]]
y = iris.target

# partition the whole dataset into training and testing dataset (30%)
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# import Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

# use entropy impurity
tree = DecisionTreeClassifier(criterion='entropy',
                              max_depth=3, random_state=0)

# training on the training dataset
tree.fit(X_train, y_train)

# import related matpoltlib package
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


# plot function  defined byself
def plot_decision_regions(X, y, classifier, test_idx=None,
                          resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='lightgray',
                    alpha=1.0, linewidth=1, marker='o',
                    s=55, label='test set')


# np.concatenate for vertical and horizontal
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined, y=y_combined,
                      classifier=tree,
                      test_idx=range(105, 150))

# Sets the label information for the coordinate axis and show the diagram
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()

# create a iris-tree.dot file to save iris decision tree model and
# use cmd command "dot -Tpng iris-tree.dot -o iris-tree.png" 
# under the same directary to visualize the iris-tree model
from sklearn.tree import export_graphviz

export_graphviz(tree, out_file='iris-tree.dot',
                feature_names=['petal length', 'petal width'])
