import numpy as np
from calc.equations import minkowski_distance

class KDNode:

    def __init__(self, indices, split_dim = None, split_value = None, left=None, right=None):
        """
        Initialize a KDNode.

        :param indices: Indices of the points in this node.
        :param split_dim: Dimension along which to split.
        :param split_value: Value at which to split.
        :param left: Left child node.
        :param right: Right child node.
        """
        self.indices = indices
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right

class KDimensionalTree:

    def __init__(self, X, leaf_size=30, metric='minkowski', p_minkowski=2):
        """
        Initialize the KDTree with the dataset X, leaf size, metric, and parameter p for Minkowski distance.

        :param X: The dataset to build the KDTree from.
        :param leaf_size: The maximum number of points in a leaf node.
        :param metric: The distance metric to use (default is 'minkowski').
        :param p: The power parameter for the Minkowski distance (default is 2).
        """
        self.X = np.array(X)
        self.leaf_size = leaf_size
        self.metric = metric
        self.p_minkowski = p_minkowski

        self.tree = self.build_tree(np.arange(self.X.shape[0]), depth=0)

    def build_tree(self, indices, depth):
        if len(indices) <= self.leaf_size:
            return KDNode(indices = indices)

        # Calculate the axis to split on
        axis = depth % self.X.shape[1]
        # Sort the indices by the values along the current axis
        sorted_indices = indices[np.argsort(self.X[indices, axis])]
        # Find the median index
        median_index = len(sorted_indices) // 2
        # Set the split dimension and value
        split_value = self.X[sorted_indices[median_index], axis]
        # Split the indices into left and right subtrees
        left_indices = self.build_tree(sorted_indices[:median_index], depth + 1)
        right_indices = self.build_tree(sorted_indices[median_index + 1:], depth + 1)

        return KDNode(indices=None, split_dim=axis, split_value=split_value, left=left_indices, right=right_indices)

    def query(self, X, k=1):
        """
        Query the KDTree for the k nearest neighbors of the point X.

        :param X: The point to query.
        :param k: The number of nearest neighbors to return.
        :return: A tuple of distances and indices of the k nearest neighbors.
        """
        X = np.asarray(X)
        return self._query(X, k)

    def _query(self, X, k):
        # Ensure X is at least 2D
        distances = []
        indices = []

        for x in X:
            # Initialize a min-heap to store the k nearest neighbors
            heap = []
            # Use a list to store the distances and indices
            self._query_point(self.tree, x, k, heap)
            # Sort the heap to get the k nearest neighbors
            heap.sort()
            # Extract distances and indices from the heap
            dists, idxs = zip(*heap[:k])
            distances.append(dists)
            indices.append(idxs)

        return np.array(distances), np.array(indices)

    def _query_point(self, node, x, k, heap):
        # Base case: if the node is a leaf node, calculate distances
        if node.indices is not None:
            # Calculate distances for all points in the leaf node
            for idx in node.indices:
                # Calculate the Minkowski distance
                dist = minkowski_distance(x, self.X[idx], p_minkowski=self.p_minkowski)
                heap.append((dist, idx))
            return

        axis = node.split_dim
        split = node.split_value

        # Check which side of the split the point is on
        if x[axis] < split:
            nearer, futher = node.left, node.right
        else:
            nearer, futher = node.right, node.left

        # Query the nearer side first
        self._query_point(nearer, x, k, heap)

        # If we have less than k points, or the furthest point in the heap is further than the split, search the further side
        if len(heap) < k or abs(x[axis] - split) < max(heap, key=lambda x: x[0])[0]:
            self._query_point(futher, x, k, heap)
