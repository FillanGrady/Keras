import keras
import keras.preprocessing.image
import time
import argparse


def compare(image, predict, correct):
    import matplotlib.pyplot as plt
    import numpy as np
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.imshow(image)

    ax2 = fig.add_subplot(122)
    color = ['b'] * 10
    color[np.amax(correct) - 1] = 'r'
    ax2.bar(x=np.arange(10), height=predict, width=.5, color=color)
    plt.show()


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
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', '-e', help="Number of epochs of training", type=int, default=50)
    args = parser.parse_args()
    data_augmentation = True
    epochs = args.epochs
    batch_size = 32
    X_train, Y_train, X_test, Y_test = get_data()
    start = time.time()
    model = keras.models.Sequential()
    model.add(keras.layers.Convolution2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(keras.layers.Convolution2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Convolution2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(keras.layers.Convolution2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if data_augmentation:
        datagen = keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        datagen.fit(X_train)
        model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), epochs=epochs, validation_data=(X_test, Y_test))
    else:
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test)
    print(score)
    print(model.metrics_names)
    print("End:%s" % (time.time() - start))
