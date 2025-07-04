import pandas as pd
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from algrithms.knn import KNeighborsClassifier
# Test with the Iris dataset

iris = load_iris()
dfIris = pd.DataFrame(iris["data"], columns=iris["feature_names"])
dfIris.loc[:, "class"] = iris["target"]

knn = KNeighborsClassifier(n_neighbors=3, algorithm='brute')

# Hold out 25% of the data for testing
xTrain, xTest, yTrain, yTest = train_test_split(dfIris.drop(columns=['class']), dfIris['class'], test_size=0.25, stratify=dfIris['class'])

knn.fit(xTrain, yTrain)
print(knn.get_params())

predictions = knn.predict(xTest)
print("Accuracy:", metrics.accuracy_score(yTest, predictions))

# Test with different algorithms and parameters
knn1 = KNeighborsClassifier(n_neighbors=3, algorithm='kd_tree')
knn1.fit(xTrain, yTrain)
predictions1 = knn1.predict(xTest)
print("Accuracy with KD Tree:", metrics.accuracy_score(yTest, predictions1))
# Test with Ball Tree
knn2 = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree')
knn2.fit(xTrain, yTrain)
predictions2 = knn2.predict(xTest)
print("Accuracy with Ball Tree:", metrics.accuracy_score(yTest, predictions2))

"""

# Get best k value with elbow method
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.use('Agg')

k_range = list(range(1, 100))
scores = []

for k in k_range:
    score_CV = cross_val_score(KNeighborsClassifier(n_neighbors=k),dfIris.drop(columns=['class']), dfIris['class'], cv=5)
    scores.append(score_CV.mean())

import matplotlib.pyplot as plt

plt.plot(k_range, scores)
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('Elbow Method for Optimal k')
plt.savefig("graph/elbow_plot.png")
print("Elbow plot saved as 'graph/elbow_plot.png'.")

# Show the best k value
best_k = k_range[scores.index(max(scores))]
print("Best k value:", best_k)

# Test model with the best k value
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(xTrain, yTrain)
predictions_best = knn_best.predict(xTest)
print("Accuracy with best k value:", metrics.accuracy_score(yTest, predictions_best))
"""