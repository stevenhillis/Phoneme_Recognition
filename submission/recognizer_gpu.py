import sys

from Recognizer_LSTM import Recognizer_LSTM


def main():
    recognizer_lstm = Recognizer_LSTM()
    print("Initialized Recognizer LSTM.")
    epochs = 15
    gpu = True
    if sys.argv[1] == 'train':
        recognizer_lstm.train(epochs, gpu)
    elif sys.argv[1] == 'test':
        model_path = sys.argv[2]
        recognizer_lstm.test(gpu, model_path)


if __name__ == '__main__':
    main()