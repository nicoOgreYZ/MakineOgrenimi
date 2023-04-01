import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Veri setini yükleme
iris = load_iris()
X = iris.data[:, :2]

# K-means modeli
k = 3
max_iter = 100
tol = 0.0001

# Başlangıç merkezlerini rastgele seçme
np.random.seed(0)
centers = X[np.random.choice(X.shape[0], k, replace=False)]

# K-means algoritması
for i in range(max_iter):
    # Küme tahminleri
    distances = np.sqrt(((X - centers[:, np.newaxis])**2).sum(axis=2))
    labels = np.argmin(distances, axis=0)

    # Küme merkezlerinin güncellenmesi
    new_centers = np.array([X[labels == j].mean(axis=0) for j in range(k)])

    # Durma koşulları
    if np.all(np.abs(centers - new_centers) < tol):
        break
    centers = new_centers

# Sonuçları görselleştirme
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=200, linewidths=3, color='r')
plt.title('K-means Sonuçları')
plt.xlabel('Çanak Yaprak Uzunluğu')
plt.ylabel('Çanak Yaprak Genişliği')
plt.show()