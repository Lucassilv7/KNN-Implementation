from typing import Literal
from collections import Counter
from algrithms.balltree import BallTree
from algrithms.kdtree import KDimensionalTree
from sklearn.base import BaseEstimator, ClassifierMixin
from calc.equations import minkowski_distance
import numpy as np

class KNeighborsClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_neighbors: int = 5, weights: Literal['uniform', 'distance'] = 'uniform',
                 algorithm: Literal['ball_tree', 'kd_tree', 'brute'] = 'brute', leaf_size: int = 30, p_minkowski: float = 2):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p_minkowski = p_minkowski

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

        if self.algorithm == 'kd_tree':
                self.tree = KDimensionalTree(self.X_train, leaf_size=self.leaf_size, metric='minkowski', p_minkowski=self.p_minkowski)
        elif self.algorithm == 'ball_tree':
            self.tree = BallTree(self.X_train, leaf_size=self.leaf_size, metric='minkowski', p_minkowski=self.p_minkowski)
        else:
            self.tree = None

    def predict(self, X):
        X = np.atleast_2d(X)  # Ensure X is at least 2D
        return [self._predict(x) for x in X]

    def _predict(self, x):

        if self.algorithm in ('kd_tree', 'ball_tree'):
            distances, indices = self.tree.query([x], k=self.n_neighbors)
            distances = distances[0]
            nearest_indices = indices[0]
        else:
            # For brute force search, calculate distances manually
            distances = [minkowski_distance(x, x_train, self.p_minkowski) for x_train in self.X_train]
            nearest_indices = np.argsort(distances)[:self.n_neighbors]

        nearest_labels = [self.y_train[i] for i in nearest_indices]

        # Apply weights if specified
        if self.weights == 'uniform':
            # Uniform weights: simply count the occurrences of each label
            label_count = Counter(nearest_labels)
            return label_count.most_common(1)[0][0]
        elif self.weights == 'distance':
            class_votes = {}
            for dist, idx in zip(distances, nearest_indices):
                label = self.y_train[idx]
                weight = 1 / dist if dist != 0 else float('inf')
                class_votes[label] = class_votes.get(label, 0) + weight
            return max(class_votes.items(), key=lambda x: x[1])[0]
        else:
            raise ValueError("Invalid value for 'weights'. Must be 'uniform' or 'distance'.")

    def get_params(self, deep=True):
        return {
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'algorithm': self.algorithm,
            'leaf_size': self.leaf_size,
            'p_minkowski': self.p_minkowski
        }