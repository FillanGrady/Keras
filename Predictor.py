import keras
import numpy as np
import argparse


def load_file(file_path=r'/home/fillan/Datasets/JoelPapers.txt'):
    with open(file_path, 'r') as f:
        text = f.read()
    return text


def save_model(model, file_name='Predictor'):
    json_file_name = file_name + ".json"
    h5_file_name = file_name + ".h5"
    model_json = model.to_json()
    with open(json_file_name, 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(h5_file_name)


def load_model(file_name='Predictor'):
    json_file_name = file_name + ".json"
    h5_file_name = file_name + ".h5"
    with open(json_file_name, 'r') as json_file:
        loaded_model = keras.models.model_from_json(json_file.read())
    loaded_model.load_weights(h5_file_name)
    return loaded_model


def create_dataset(dataset, look_back=1):
    datset_as_list = list(dataset)
    temp = np.zeros(shape=(128, len(datset_as_list)))
    for i, c in enumerate(datset_as_list):
        num_form = ord(c)
        if num_form < 128:
            temp[num_form, i] = 1
        else:
            temp[32, i] = 1  # This is equal to " "
    return np.array(temp[:, :-look_back]), np.array(temp[:, look_back:])

def create_output(model, starting_character="S", periods=1):



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', '-e', help="Number of epochs of training", type=int, default=100)
    parser.add_argument('-look_back', '-l', help="Number of periods to look back on", type=int, default=5)
    parser.add_argument('-save', '-s', help="Save location", type=str, default=None)
    parser.add_argument('-load', '-o', help="Load location", type=str, default=None)
    parser.add_argument('-periods', '-p', help="Number of periods in output file", type=int, default=1)
    args = parser.parse_args()
    text = load_file()
    X, Y = create_dataset(text)
    if args.load is None:
        model = keras.Sequential()
        model.add(keras.layers.LSTM(units=20, input_shape=(1, 1)))
        model.add(keras.layers.Dense(units=1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X, Y, batch_size=1, epochs=args.epochs, verbose=2)
    else:
        model = load_model(args.load)
    if args.save is not None:
        save_model(model, file_name=args.save)