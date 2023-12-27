from preprocessing import Preprocess
from pathlib import Path
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, LSTM, Dense, Embedding, Masking, RepeatVector, TimeDistributed, GRU, Dropout
import numpy as np
from keras.utils import to_categorical

path = Path(__file__).resolve()

eng_path = path.parent.parent / "data" / "small_vocab_en.txt"
fr_path = path.parent.parent / "data" / "small_vocab_fr.txt"
json = path.parent / "data.json"


# def model(vocabSize, embedDimensions, hiddenUnits):
#     """
#     Build and train a RNN model using word embedding on x and y
#     :param input_shape: Tuple of input shape
#     :param output_sequence_length: Length of output sequence
#     :param english_vocab_size: Number of unique English words in the dataset
#     :param french_vocab_size: Number of unique French words in the dataset
#     :return: Keras model built, but not trained
#     """
#     # TODO: Implement

#     # Hyperparameters
#     learning_rate = 0.005

#     # TODO: Build the layers
#     model = Sequential()
#     model.add(Embedding(vocabSize, 256, input_length=input_shape[1], input_shape=input_shape[1:]))
#     model.add(GRU(256, return_sequences=True))
#     model.add(TimeDistributed(Dense(1024, activation='relu')))
#     model.add(Dropout(0.5))
#     model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax')))

#     # Compile model
#     model.compile(loss="sparse_categorical_crossentropy",
#                   optimizer="adam",
#                   metrics=['accuracy'])
#     return model


def simple_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a basic RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    model = Sequential()
    model.add(GRU(256, input_shape=input_shape[1:], return_sequences=True))
    model.add(TimeDistributed(Dense(1024, activation="relu")))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(french_vocab_size, activation="softmax")))

    # Compile model
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def main():
    BATCH_SIZE = 128
    Preprocess(eng_path, fr_path, BATCH_SIZE)
    (
        sourceVectors,
        targetVectors,
        sourcePadded,
        targetPadded,
        sourceTokenToIndex,
        sourceIndexToToken,
        targetTokenToIndex,
        targetIndexToToken,
    ) = Preprocess.loadPreprocessedData(json)

    vocabSizeSRC = len(sourceVectors)
    vocabSizeTRG = len(targetVectors)
    print(max(len(targetPadded)))
    print(max(len(sourcePadded)))
    # max_length = 
    embedDimensions = 64
    hiddenUnits = 128

    modelLSTM = simple_model(vocabSize, vocabSizeSRC, vocabSizeTRG)
    modelLSTM.summary()

    targetPadded_onehot = [to_categorical(seq, num_classes=vocabSize) for seq in targetPadded]
    print("boop")

    modelLSTM.fit(
        [sourcePadded[:10000], targetPadded[:10000]], targetPadded_onehot[:10000], batch_size=BATCH_SIZE, epochs=2
    )
    evals = modelLSTM.evaluate(sourcePadded[:2000], targetPadded[:2000])
    acc = evals[1]
    print(acc)


if __name__ == "__main__":
    main()
