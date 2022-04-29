import csv
import numpy as np


class MovieDataset:
    def __init__(
        self,
        data_file,
        data_size,
        text_rpst="ngram",
        gram_num=3,
        test_rate = 0.3
    ):
        self.text_rpst = text_rpst
        self.gram_num = gram_num
        self.category = dict() # {category:index}
        self.vocab = dict() # {word:index}
        self.train_x, self.train_y, self.test_x, self.test_y = self._data_process(data_file, data_size, test_rate)

    def _data_read(self, data_file, data_size):
        with open(data_file) as f:
            tsvreader = csv.reader(f, delimiter='\t')
            data = list(tsvreader)
        data = data[1:data_size] # remove the header

        return data

    def _dict_update(self, data):
        "Update the dictionary of category and vocabulary."
        if self.text_rpst == "bow":
            for item in data:
                if item[3] not in self.category:
                    self.category[item[3]] = len(self.category) # hash
                words = item[2].lower().split()
                for word in words:
                    if word not in self.vocab:
                        self.vocab[word] = len(self.vocab) # hash
        elif self.text_rpst == "ngram":
            for n in range(1, self.gram_num + 1):
                for item in data:
                    if item[3] not in self.category:
                        self.category[item[3]] = len(self.category)
                    words = item[2].lower().split()
                    for i in range(len(words) - n + 1):
                        word = words[i:i + n]
                        word = "_".join(word) # e.g., i_like is a brand new "word"
                        if word not in self.vocab:
                            self.vocab[word] = len(self.vocab)
        else:
            raise Exception("Unknown text representation!")

    def _data_split(self, data, test_rate):
        train = list()
        test = list()
        for item in data:
            if np.random.random_sample() > test_rate:
                train.append(item)
            else:
                test.append(item)

        return train, test

    def _data_vectorization(self, data):
        x = np.zeros((len(data), len(self.vocab)))
        if self.text_rpst == "bow":
            for i in range(len(data)):
                words = data[i][2].lower().split()
                for word in words:
                    x[i][self.vocab[word]] = 1
        elif self.text_rpst == "ngram":
            for n in range(1, self.gram_num):
                for i in range(len(data)):
                    words = data[i][2].lower().split()
                    for j in range(len(words) - n + 1):
                        word = words[j:j + n]
                        word = "_".join(word)
                        x[i][self.vocab[word]] = 1
        y = [int(item[3]) for item in data]

        return x, y

    def _data_process(self, data_file, data_size, test_rate):
        data = self._data_read(data_file, data_size)
        self._dict_update(data)
        train, test = self._data_split(data, test_rate)
        train_x, train_y = self._data_vectorization(train)
        test_x, test_y = self._data_vectorization(test)

        return train_x, train_y, test_x, test_y