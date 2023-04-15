import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class SinirAgi:
    def __init__(self):
        # Veri setini yükle
        iris = load_iris()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            iris.data, iris.target, test_size=0.2)
        self.built = True

        # Modeli oluştur
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])

        # Modeli derle
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self):
        # Modeli eğit
        self.model.fit(self.X_train, self.y_train, epochs=100)

    def evaluate(self):
        # Modeli değerlendir
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        print(f"Kayıp: {loss:.4f}")
        print(f"Doğruluk: {accuracy:.4f}")

    def predict(self, data):
        # Tahmin yap
        predictions = self.model.predict(data)
        return predictions.argmax(axis=1)
    
if __name__=='__main__':
    sa = SinirAgi()
    sa.train()
    sa.evaluate()

