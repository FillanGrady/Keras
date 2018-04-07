import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
import keras

np.random.seed(123)


def get_data():
    ((X_train, Y_train), (X_test, Y_test)) = keras.datasets.mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    Y_train = keras.utils.np_utils.to_categorical(Y_train, 10)
    Y_test = keras.utils.np_utils.to_categorical(Y_test, 10)
    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = get_data()
    model = keras.models.Sequential()
    model.add(keras.layers.Convolution2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(keras.layers.Convolution2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=32, nb_epoch=5, verbose=1)
    score = model.evaluate(X_test, Y_test, verbose=0)
    print(score)