# K-Means : finde claster of sampl

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

iris = load_iris()

# K-Means 
kmn = KMeans(n_clusters=3)
kmn.fit(iris.data)

# Predict
labels = kmn.predict(iris.data)

# Centers 
centroids = kmn.cluster_centers_

# Plot
plt.scatter(iris.data[:,0], iris.data[:,2], c=labels)
plt.scatter(centroids[:,0], centroids[:,2], marker='x', s=150)
