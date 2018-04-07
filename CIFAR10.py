import keras

def get_data():
    ((X_train, Y_train), (X_test, Y_test)) = keras.datasets.cifar10.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    Y_train = keras.utils.np_utils.to_categorical(Y_train, 10)
    Y_test = keras.utils.np_utils.to_categorical(Y_test, 10)
    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = get_data()
    print("X_train shape: %s" % str(X_train.shape))
    print("Y_train shape: %s" % str(Y_train.shape))
    print("X_test shape: %s" % str(X_test.shape))
    print("Y_test shape: %s" % str(Y_test.shape))
    model = keras.models.Sequential()
    model.add(keras.layers.Convolution2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(keras.layers.Convolution2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.metrics_names)
    model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)
    score = model.evaluate(X_test, Y_test)
    print(score)
