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


def list2vec(list):
    temp = np.zeros(shape=(len(list), 128))
    for i, c in enumerate(list):
        num_form = ord(c)
        if num_form < 128:
            temp[i, num_form] = 1
        else:
            temp[i, 32] = 1  # This is equal to " "
    return np.array(temp)


def vec2str(vector):
    s = vector.argmax(axis=1)
    output = ""
    for i in range(np.size(s)):
        output += chr(s[i])
    return output


def create_dataset(dataset, look_back=1):
    datset_as_vec = list2vec(list(dataset))
    Y = datset_as_vec[look_back:, :]
    X = np.zeros((np.shape(datset_as_vec)[0] - look_back, look_back, 128))
    for i in range(np.shape(X)[0]):
        X[i, :, :] = datset_as_vec[i:i + look_back, :].reshape(look_back, 128)
    return X, Y


def create_output(model, look_back, total_periods_left=1):
    sequence = list2vec(("hello " * 10)[:look_back]).reshape(1, look_back, 128)
    output_list = []
    while total_periods_left > 0:
        raw_output = model.predict(sequence)[0]
        maxed_output = np.zeros(128, dtype=np.bool)
        maxed_output[raw_output.argmax()] = True
        output_list.append(maxed_output)
        if maxed_output[46] is True:
            total_periods_left -= 1
        sequence = np.roll(sequence, shift=-1, axis=1)
        sequence[0, -1, :] = maxed_output
    return vec2str(np.array(output_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-file", "-f", help="File to load", type=str, default=r'/home/fillan/Datasets/JoelPapers.txt')
    parser.add_argument('-epochs', '-e', help="Number of epochs of training", type=int, default=100)
    parser.add_argument('-look_back', '-l', help="Number of periods to look back on", type=int, default=5)
    parser.add_argument('-save', '-s', help="Save location", type=str, default=None)
    parser.add_argument('-load', '-o', help="Load location", type=str, default=None)
    parser.add_argument('-periods', '-p', help="Number of periods in output file", type=int, default=1)
    args = parser.parse_args()
    text = load_file(args.file)
    X, Y = create_dataset(text, args.look_back)
    print("Finished converting dataset")
    if args.load is None:
        model = keras.Sequential()
        model.add(keras.layers.LSTM(units=50, input_shape=(args.look_back, 128)))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(units=128))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X, Y, batch_size=1, epochs=args.epochs, verbose=1)
    else:
        model = load_model(args.load)
    if args.save is not None:
        save_model(model, file_name=args.save)
    print(create_output(model, look_back=args.look_back, total_periods_left=3))
