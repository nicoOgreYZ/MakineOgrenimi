import numpy as np

def h(X, theta):
    """
    Tahmin fonksiyonu h(x) = theta0 + theta1*x1
    """
    return np.dot(X, theta)

def compute_cost(X, y, theta):
    """
    Gider fonksiyonu J(theta) = (1/2*m)*sum((h(x) - y)^2)
    """
    m = len(y)
    J = (1/(2*m)) * np.sum((h(X, theta) - y) ** 2)
    return J

def gradient_descent(X, y, theta, alpha, iterations):
    """
    Gradyan iniş algoritması
    """
    m = len(y)
    J_history = np.zeros(iterations)
    
    for i in range(iterations):
        theta = theta - (alpha/m) * np.dot(X.T, (h(X, theta) - y))
        J_history[i] = compute_cost(X, y, theta)
    
    return theta, J_history

# Örnek veri kümesi
X = np.array([[1, 5], [1, 7], [1, 8], [1, 10], [1, 12], [1, 15]])
y = np.array([11, 13, 15, 19, 21, 25])

# Theta parametreleri
theta = np.zeros(2)

# Gradyan iniş hiperparametreleri
alpha = 0.01
iterations = 1000

# Gradyan iniş uygulama
theta, J_history = gradient_descent(X, y, theta, alpha, iterations)

# Eğim ve sabit katsayıları yazdırma
print("Eğim katsayısı: ", theta[1])
print("Sabit katsayı: ", theta[0])

# Gider fonksiyonunun zaman içindeki değişimini gösteren grafiği çizdirme
import matplotlib.pyplot as plt
plt.plot(J_history)
plt.xlabel('İterasyonlar')
plt.ylabel('Gider')
plt.show()
