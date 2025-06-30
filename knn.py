from typing import Literal
from collections import Counter
from sklearn.neighbors import BallTree, KDTree
from sklearn.base import BaseEstimator, ClassifierMixin
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
            self.tree = KDTree(self.X_train, leaf_size=self.leaf_size, metric='minkowski', p=self.p_minkowski)
        elif self.algorithm == 'ball_tree':
            self.tree = BallTree(self.X_train, leaf_size=self.leaf_size, metric='minkowski', p=self.p_minkowski)
        else:
            self.tree = None

    def predict(self, X):
        X = np.atleast_2d(X)  # Ensure X is at least 2D
        return [self._predict(x) for x in X]

    def _predict(self, x):

        if self.algorithm in ('kd_tree', 'ball_tree'):
            distances, indices = self.tree.query(x, k=self.n_neighbors)
            distances = distances[0]
            nearest_indices = indices[0]
        else:
            # For brute force search, calculate distances manually
            distances = [self.minkowski_distance(x, x_train) for x_train in self.X_train]
            nearest_indices = np.argsort(distances)[:self.n_neighbors]

        nearest_labels = [self.y_train[i] for i in nearest_indices]

        # Apply weights if specified
        if self.weights == 'uniform':
            # Uniform weights: simply count the occurrences of each label
            label_count = Counter(nearest_labels)
            return label_count.most_common(1)[0][0]
        elif self.weights == 'distance':
            # Distance weights: weight labels by the inverse of their distances
            class_votes = {}
            for i in nearest_indices:
                label = self.y_train[i]
                weight = 1 / distances[i] if distances[i] != 0 else float('inf')  # Avoid division by zero
                class_votes[label] = class_votes.get(label, 0) + weight
            # Return the label with the highest weighted vote
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

    def minkowski_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2) ** self.p_minkowski) ** (1 / self.p_minkowski)

    def score(self, X, y):
        predictions = self.predict(X)
        correct_predictions = sum(pred == true for pred, true in zip(predictions, y))
        return correct_predictions / len(y) if len(y) > 0 else 0.0