import json
import numpy as np
from pathlib import Path
import os
import copy
from keras.preprocessing.sequence import pad_sequences


# initializing path variable
path = Path(__file__).resolve()

# getting paths to data
eng_path = path.parent.parent / "data/small_vocab_en.txt"
fr_path = path.parent.parent / "data/small_vocab_fr.txt"


class Preprocess:
    def __init__(self, src_path, trg_path, batch_size):
        # load files into variables
        srcText = Preprocess.loadData(src_path)
        trgText = Preprocess.loadData(trg_path)

        # ensures no duplicate tokens due to capitalization
        srcText = srcText.lower()
        trgText = trgText.lower()

        # get index dictionaries
        srcTokenToIndex, srcIndexToToken = Preprocess.createVocab(srcText)
        trgTokenToIndex, trgIndexToToken = Preprocess.createVocab(trgText)

        # get numeric token vectors
        srcVectors, trgVectors = Preprocess.sentToVect(
            srcText,
            trgText,
            srcTokenToIndex,
            trgTokenToIndex,
        )

        sourcePadded, targetPadded = Preprocess.padBatches(
            srcVectors,
            trgVectors,
            srcTokenToIndex,
            # batch_size,
            # round(len(srcVectors) / batch_size),
        )

        # get data into dictionary
        data = {
            "sourceVectors": srcVectors,
            "targetVectors": trgVectors,
            "sourcePadded": sourcePadded,
            "targetPadded": targetPadded,
            "sourceTokenToIndex": srcTokenToIndex,
            "sourceIndexToToken": srcIndexToToken,
            "targetTokenToIndex": trgTokenToIndex,
            "targetIndexToToken": trgIndexToToken,
        }

        # save data to json file for future use
        with open(path.parent / "data.json", "w") as file:
            json.dump(data, file)

    # function to get data from file
    def loadData(path):
        file = os.path.join(path)
        with open(file, "r", encoding="utf-8") as f:
            data = f.read()

        return data

    # function to load preprocessed data from json saved file
    def loadPreprocessedData(path):
        with open(path, "r") as file:
            data = json.load(file)

        # extract data
        sourceVectors = data["sourceVectors"]
        targetVectors = data["targetVectors"]
        sourcePadded = data["sourcePadded"]
        targetPadded = data["targetPadded"]

        # unjsonify padded vectors
        for i in range(len(data["sourcePadded"])):
            data["sourcePadded"][i] = np.array(data["sourcePadded"][i])
            data["targetPadded"][i] = np.array(data["targetPadded"][i])
            data["targetPadded"][i] = data["targetPadded"][i].reshape(*data["targetPadded"][i], 1)

        sourceTokenToIndex = data["sourceTokenToIndex"]
        sourceIndexToToken = data["sourceIndexToToken"]
        targetTokenToIndex = data["targetTokenToIndex"]
        targetIndexToToken = data["targetIndexToToken"]

        return (
            sourceVectors,
            targetVectors,
            sourcePadded,
            targetPadded,
            sourceTokenToIndex,
            sourceIndexToToken,
            targetTokenToIndex,
            targetIndexToToken,
        )

    # gets set of unique tokens, dictionary of tokens to indicies and the inverse
    def createVocab(text):
        # special tokens to add to sentences that most NLP models require.
        # <PAD> is added to pad sentences to the same length,
        # <EOS> represents the end of sentence, added to end of sentence,
        # <UNK> is used for unknown tokens, which are tokens that the machine has not seen in fitting,
        # <GO> is placed at the start of the target sentence, signalling to the model that this is the start of
        # the translation.
        SPECIAL_TOKENS = {"<PAD>": 0, "<EOS>": 1, "<UNK>": 2, "<GO>": 3}

        # set automatically ignores duplicates, perfect for getting unique tokens
        vocab = set(text.split())

        # initialize token to index dictionary, place special tokens at the beginning
        tokenToIndex = copy.copy(SPECIAL_TOKENS)
        for index, token in enumerate(vocab, len(SPECIAL_TOKENS)):
            tokenToIndex[token] = index

        indexToToken = {index: token for index, token in tokenToIndex.items()}

        return tokenToIndex, indexToToken

    # turns sentences into vectors of their respective token indices
    def sentToVect(srcText, trgText, srcTokenToIndex, trgTokenToIndex):
        srcVect = []
        trgVect = []

        # split text into sentences
        srcSent = srcText.split("\n")
        trgSent = trgText.split("\n")

        for i in range(len(srcSent)):
            # get current sentence
            curSrcSent = srcSent[i]
            curTrgSent = trgSent[i]

            # split into tokens
            srcTokens = curSrcSent.split(" ")
            trgTokens = curTrgSent.split(" ")

            # vectors of token indices
            srcTokenVect = []
            trgTokenVect = []

            # iterates through sentence, adding indicies to vector
            for i, token in enumerate(srcTokens):
                # ensures token isn't null
                if token != "":
                    srcTokenVect.append(srcTokenToIndex[token])

            for i, token in enumerate(trgTokens):
                # ensures token isn't null
                if token != "":
                    trgTokenVect.append(trgTokenToIndex[token])

            # add end of sentence token to end of target sentence
            trgTokenVect.append(trgTokenToIndex["<EOS>"])

            # append token vectors to sentence vectors
            srcVect.append(srcTokenVect)
            trgVect.append(trgTokenVect)

        return srcVect, trgVect

    def padBatches(src, trg, tokenToIndex):
        srcPadded = []
        trgPadded = []

        trg = [[tokenToIndex["<GO>"]] + seq for seq in trg]

        srcLongest = max(len(src[j]) for j in range(len(src)))
        trgLongest = max(len(trg[j]) for j in range(len(trg)))
        # get longest of the two
        if srcLongest > trgLongest:
            maxLength = srcLongest
        else:
            maxLength = trgLongest

        for sentence in src:
            curLength = len(sentence)
            while curLength < maxLength:
                sentence.append(tokenToIndex["<PAD>"])
                curLength += 1
            srcPadded.append(sentence)

            for sentence in trg:
                curLength = len(sentence)
                while curLength < maxLength:
                    sentence.append(tokenToIndex["<PAD>"])
                    curLength += 1
                trgPadded.append(sentence)

        return srcPadded, trgPadded

    # pads sentences based on longest sentence within the batch
    # def padBatches(src, trg, tokenToIndex, batchSize, numBatches):
    #     srcPadded = []
    #     trgPadded = []

    #     # loop for however many batches there are
    #     for i in range(numBatches):
    #         srcPaddedBatch = []
    #         trgPaddedBatch = []

    #         # get current set of sentences based on batch size
    #         curSrcBatch = src[(batchSize * i) : (batchSize * (i + 1))]
    #         curTrgBatch = trg[(batchSize * i) : (batchSize * (i + 1))]

    #         # adding special token to target sentences for decoder
    #         curTrgBatch = [[tokenToIndex["<GO>"]] + seq for seq in curTrgBatch]
            
    #         # get longest sentence for current batch of source and target sentences
    #         srcLongest = max(len(curSrcBatch[j]) for j in range(batchSize))
    #         trgLongest = max(len(curTrgBatch[j]) for j in range(batchSize))

    #         # get longest of the two
    #         if srcLongest > trgLongest:
    #             maxLength = srcLongest
    #         else:
    #             maxLength = trgLongest

    #         # pad sentences until theyre all equal lengths to the longest sentence in the batch
    #         for sentence in curSrcBatch:
    #             curLength = len(sentence)
    #             while curLength < maxLength:
    #                 sentence.append(tokenToIndex["<PAD>"])
    #                 curLength += 1
    #             srcPaddedBatch.append(sentence)

    #         for sentence in curTrgBatch:
    #             curLength = len(sentence)
    #             while curLength < maxLength:
    #                 sentence.append(tokenToIndex["<PAD>"])
    #                 curLength += 1
    #             trgPaddedBatch.append(sentence)

    #         srcPadded.append(srcPaddedBatch)
    #         trgPadded.append(trgPaddedBatch)

    #     return srcPadded, trgPadded


def main():
    BATCH_SIZE = 32
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
    ) = Preprocess.loadPreprocessedData(path.parent / "data.json")


if __name__ == "__main__":
    main()
