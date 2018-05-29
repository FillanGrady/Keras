import Predictor
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-file", "-f", help="File to load", type=str, default=r'/home/fillan/Datasets/JoelPapersTest.txt')
    parser.add_argument('-log', help='Log save location', type=str, default=None)
    parser.add_argument('-look_back', '-l', help="Number of periods to look back on", type=int, default=5)
    parser.add_argument('-load', '-o', help="Load location", type=str, default=None)
    args = parser.parse_args()
    text = Predictor.load_file(args.file)
    X, Y = Predictor.create_dataset(text, args.look_back)
    model = Predictor.load_model(args.load)
    model.compile(loss='mean_squared_error', optimizer='adam')
    metrics = model.evaluate(X, Y)
    with open(args.log, 'w+') as f:
        f.write(os.linesep + "Test Set" + os.linesep)
        f.write(str(metrics))