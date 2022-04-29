import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class Movie(Dataset):
    def __init__(self, sentences, emotions):
        self.sentences = sentences
        self.emotions = emotions

    def __getitem__(self, item):
        return self.sentences[item], self.emotions[item]

    def __len__(self):
        return len(self.emotions)

class MovieDataset:

    PAD = 0
    pad_token = "#pad#"

    def __init__(self, data_file, data_size, test_rate):
        self.max_sent_len = 0
        self.category = dict() # {category:index}
        self.vocab = dict() # {word:index}
        self.vocab[self.pad_token] = self.PAD

        self.train, self.test = self._data_process(data_file, data_size, test_rate)

    def _data_read(self, data_file, data_size):
        with open(data_file) as f:
            tsvreader = csv.reader(f, delimiter='\t')
            data = list(tsvreader)
        data = data[1:data_size]
        data.sort(key=lambda x:len(x[2].split())) # sort from short to long, then it's easy to keep the same length in one batch
        self.max_sent_len = len(data[-1][2].split()) # length of the longest sentence

        return data

    def _dict_update(self, data):
        for item in data:
            if item[3] not in self.category:
                self.category[item[3]] = len(self.category) # hash
            words = item[2].lower().split()
            for word in words:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab) # hash

    def _data_split(self, data, test_rate):
        train = list()
        test = list()
        for item in data:
            if np.random.random() > test_rate:
                train.append(item)
            else:
                test.append(item)

        return train, test

    def _data_vectorization(self, data):
        x = list()
        for item in data:
            words = item[2].lower().split()
            word_indices = [self.vocab[word] for word in words] # map each sentence to a list of index in the vocabulary
            x.append(word_indices)
        y = [int(item[3]) for item in data]

        return Movie(x, y) # package

    def _data_process(self, data_file, data_size, test_rate):
        data = self._data_read(data_file, data_size)
        self._dict_update(data)
        train, test = self._data_split(data, test_rate)
        train = self._data_vectorization(train)
        test = self._data_vectorization(test)

        return train, test

    def collate_fn(self, batch_data):
        sentences, emotions = zip(*batch_data)
        sentences = [torch.LongTensor(sentence) for sentence in sentences]
        padded_sents = pad_sequence(sentences, batch_first=True, padding_value=self.PAD) # keep the same length in one batch
        return torch.LongTensor(padded_sents), torch.LongTensor(emotions)