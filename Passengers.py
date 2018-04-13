import keras
import pandas
import matplotlib.pyplot as plt
import numpy as np
import argparse


def load_file(file_path = r'/home/fillan/Datasets/international-airline-passengers.csv'):
    dataset = pandas.read_csv(filepath_or_buffer=file_path, usecols=[1], engine='python', skipfooter=3)
    arr = np.array(dataset.values.astype('float32'))
    arr /= np.max(arr)
    divider = arr.size * 2 // 3
    return arr[:divider], arr[divider:]


def create_dataset(dataset, look_back=1):
    Y = dataset[look_back:]
    X = np.zeros((np.size(dataset) - look_back, 1, look_back))
    for i in range(np.size(dataset) - look_back):
        X[i, :, :] = dataset[i:i + look_back].reshape(1, look_back)
    return X, Y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', '-e', help="Number of epochs of training", type=int, default=100)
    parser.add_argument('-look_back', '-l', help="Number of periods to look back on", type=int, default=5)
    args = parser.parse_args()
    train, test = load_file()
    trainX, trainY = create_dataset(dataset=train, look_back=args.look_back)
    testX, testY = create_dataset(dataset=test, look_back=args.look_back)
    model = keras.Sequential()
    model.add(keras.layers.LSTM(units=4, input_shape=(1, args.look_back)))
    model.add(keras.layers.Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, batch_size=1, epochs=args.epochs, verbose=2)
    testPredict = model.predict(testX)
    plt.plot(trainY)
    plt.plot(np.array(range(train.size, train.size + test.size - args.look_back)), testPredict)
    plt.plot(np.array(range(train.size, train.size + test.size - args.look_back)), testY)
    plt.savefig("Passengers.png")