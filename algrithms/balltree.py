import numpy as np
from calc.equations import minkowski_distance

class BallNode:

    def __init__(self, indices, center=None, radius=None, left=None, right=None):
        """
        Initialize a BallNode.

        :param indices: Indices of the points in this node.
        :param center: Center of the ball.
        :param radius: Radius of the ball.
        :param left: Left child node.
        :param right: Right child node.
        """
        self.indices = indices
        self.center = center
        self.radius = radius
        self.left = left
        self.right = right

class BallTree:

    def __init__(self, X, leaf_size=30, metric='minkowski', p_minkowski=2):
        """
        Initialize the BallTree with the dataset X, leaf size, metric, and parameter p for Minkowski distance.

        :param X: The dataset to build the BallTree from.
        :param leaf_size: The maximum number of points in a leaf node.
        :param metric: The distance metric to use (default is 'minkowski').
        :param p: The power parameter for the Minkowski distance (default is 2).
        """
        self.X = np.asarray(X)
        self.leaf_size = leaf_size
        self.metric = metric
        self.p_minkowski = p_minkowski

        self.tree = self.build_tree(np.arange(self.X.shape[0]))

    def _compute_center_radius(self, indices):
        """
        Compute the center and radius of the ball for the given indices.

        :param indices: Indices of the points in this node.
        :return: Center and radius of the ball.
        """
        points = self.X[indices]
        center = np.mean(points, axis=0)
        radius = np.max([minkowski_distance(center, point, p_minkowski = self.p_minkowski) for point in points])
        return center, radius

    def build_tree(self, indices):

        # 
        center, radius = self._compute_center_radius(indices)

        if len(indices) <= self.leaf_size:
            return BallNode(indices=indices, center=center, radius=radius)

        pts = self.X[indices]
        dists = np.linalg.norm(pts - center, axis=1)
        pivot1_idx = np.argmax(dists)
        pivot1 = pts[pivot1_idx]
        dists2 = np.linalg.norm(pts - pivot1, axis=1)
        pivot2_idx = np.argmax(dists2)
        pivot2 = pts[pivot2_idx]

        # Split the indices into left and right subtrees based on the pivot points
        left_indices = []
        right_indices = []
        for idx in indices:
            distance1 = minkowski_distance(self.X[idx], pivot1, p_minkowski=self.p_minkowski)
            distance2 = minkowski_distance(self.X[idx], pivot2, p_minkowski=self.p_minkowski)
            if distance1 < distance2:
                left_indices.append(idx)
            else:
                right_indices.append(idx)

        if len(left_indices) == 0 or len(right_indices) == 0:
            return BallNode(indices=indices, center=center, radius=radius)

        left = self.build_tree(np.array(left_indices))
        right = self.build_tree(np.array(right_indices))

        return BallNode(indices=None, center=center, radius=radius, left=left, right=right)

    def query(self, X, k=1):
        """
        Query the BallTree for the k nearest neighbors of the point X.

        :param X: The point to query.
        :param k: The number of nearest neighbors to return.
        :return: A tuple of distances and indices of the k nearest neighbors.
        """
        X = np.asarray(X)
        return self._query(X, k)

    def _query(self, X, k):
        distances = []
        indices = []

        for x in X:
            heap = []
            self._query_point(self.tree, x, k, heap)
            heap.sort()
            dists, idxs = zip(*heap[:k])
            distances.append(dists)
            indices.append(idxs)

        return np.array(distances), np.array(indices)

    def _query_point(self, node, x, k, heap):
        if node.indices is not None:
            for idx in node.indices:
                dist = minkowski_distance(x, self.X[idx], p_minkowski=self.p_minkowski)
                heap.append((dist, idx))
            return

        d_left = minkowski_distance(x, node.left.center, p_minkowski=self.p_minkowski) if node.left else float('inf')
        d_right = minkowski_distance(x, node.right.center, p_minkowski=self.p_minkowski) if node.right else float('inf')

        if d_left < d_right:
            first, second = node.left, node.right
        else:
            first, second = node.right, node.left

        if first is not None:
            self._query_point(first, x, k, heap)

        if second is not None and second.center is not None and second.radius is not None:
            best_so_far = max(heap, key=lambda item: item[0])[0] if heap else float('inf')
            dist_to_second = minkowski_distance(x, second.center, p_minkowski=self.p_minkowski)
            if dist_to_second - second.radius < best_so_far:
                self._query_point(second, x, k, heap)
