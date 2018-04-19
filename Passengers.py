import keras
import pandas
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide messy TensorFlow warnings


def load_file(file_path = r'/home/fillan/Datasets/international-airline-passengers.csv'):
    dataset = pandas.read_csv(filepath_or_buffer=file_path, usecols=[1], engine='python', skipfooter=3)
    arr = np.array(dataset.values.astype('float32'))
    arr /= np.max(arr)
    divider = arr.size * 2 // 3
    return arr[:divider], arr[divider:]


def create_dataset(dataset, look_back=1):
    Y = dataset[look_back:]
    X = np.zeros((np.size(dataset) - look_back, look_back, 1))
    for i in range(np.size(dataset) - look_back):
        X[i, :, :] = dataset[i:i + look_back].reshape(look_back, 1)
    return X, Y


def save_model(model, file_name='Passengers'):
    json_file_name = file_name + ".json"
    h5_file_name = file_name + ".h5"
    model_json = model.to_json()
    with open(json_file_name, 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(h5_file_name)


def load_model(file_name='Passengers'):
    json_file_name = file_name + ".json"
    h5_file_name = file_name + ".h5"
    loaded_model = None
    with open(json_file_name, 'r') as json_file:
        loaded_model = keras.models.model_from_json(json_file.read())
    loaded_model.load_weights(h5_file_name)
    return loaded_model


def predict(model, initial_sequence, num_iterations):
    output = []
    for i in range(num_iterations):
        o = model.predict(initial_sequence)[0][0]
        initial_sequence = np.roll(initial_sequence, shift=-1, axis=1)
        initial_sequence[0, -1, 0] = o
        output.append(o)
    return np.array(output)


def pretty_print(arr):
    line = ""
    for i in range(arr.shape[1]):
        line += "%.2f" % arr[0][i][0] + " "
    print(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', '-e', help="Number of epochs of training", type=int, default=10)
    parser.add_argument('-look_back', '-l', help="Number of periods to look back on", type=int, default=1)
    parser.add_argument('-save', '-s', help="Save location", type=str, default=None)
    parser.add_argument('-load', '-o', help="Load location", type=str, default=None)
    parser.add_argument('-savefig', '-f', help="Save figure?", action="store_true")
    args = parser.parse_args()
    train, test = load_file()
    trainX, trainY = create_dataset(dataset=train, look_back=args.look_back)
    testX, testY = create_dataset(dataset=test, look_back=args.look_back)
    model = None
    if args.load is None:
        model = keras.Sequential()
        model.add(keras.layers.LSTM(units=50, input_shape=(args.look_back, 1), return_sequences=True))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.LSTM(units=100, return_sequences=False))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(units=1))
        model.compile(loss='mean_squared_error', optimizer='rmsprop')
        model.fit(trainX, trainY, batch_size=1, epochs=args.epochs, verbose=2)
    else:
        model = load_model(args.load)
    testPredict = predict(model, initial_sequence=trainX[-1:, :, :], num_iterations=testY.size)
    plt.plot(trainY)
    plt.plot(np.array(range(train.size, train.size + test.size - args.look_back)), testPredict)
    plt.plot(np.array(range(train.size, train.size + test.size - args.look_back)), testY)
    if args.savefig:
        matplotlib.use('Agg')
        plt.savefig("Passengers.png")
    else:
        plt.show()
    if args.save is not None:
        save_model(model, file_name=args.save)