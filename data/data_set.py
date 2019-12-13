from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds

def get_data():
    ((X_train, y_train), (X_test, y_test)) = fashion_mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype("float32") / 255.
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype("float32") / 255.
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return ((X_train, y_train), (X_test, y_test))
