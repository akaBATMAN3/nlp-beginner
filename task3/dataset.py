import jsonlines
import re
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class SNLI(Dataset):
    def __init__(self, sents1, sents2, relations):
        self.sents1 = sents1
        self.sents2 = sents2
        self.relations = relations

    def __getitem__(self, item):
        return self.sents1[item], self.sents2[item], self.relations[item]

    def __len__(self):
        return len(self.relations)

class SNLIDataset:

    PAD = 0
    pad_token = "#pad#"

    def __init__(self, data_file, data_size, type_dict):
        self.type_dict = type_dict
        self.vocab = dict()
        self.vocab[self.pad_token] = self.PAD
        self.max_sent_len = 0

        self.train = self._data_process(data_file["train"], data_size["train"])
        self.val = self._data_process(data_file["val"], data_size["val"])
        self.test = self._data_process(data_file["test"], data_size["test"])

    def _data_read(self, data_file, data_size):
        data = list()
        count = 0
        with jsonlines.open(data_file, mode='r') as reader:
            for line in reader: # line is dict
                data.append([line["sentence1"], line["sentence2"], line["gold_label"]])
                count += 1
                if count == data_size:
                    break
        #data.sort(key=lambda x:len(x[0].split()))

        return data

    def _vocab_update(self, data):
        pattern = '[A-Za-z|\']+'
        for item in data:
            for i in range(2):
                words = re.findall(pattern, item[i].lower()) # word segmentation
                for word in words:
                    if word not in self.vocab:
                        self.vocab[word] = len(self.vocab) # hash

    def _data_vectorization(self, data):
        x1 = list()
        x2 = list()
        pattern = '[A-Za-z|\']+'
        for item in data:
            words = re.findall(pattern, item[0].lower())
            word_indices = [self.vocab[word] for word in words]
            self.max_sent_len = max(self.max_sent_len, len(word_indices))
            x1.append(word_indices)

            words = re.findall(pattern, item[1].lower())
            word_indices = [self.vocab[word] for word in words]
            self.max_sent_len = max(self.max_sent_len, len(word_indices))
            x2.append(word_indices)
        y = [self.type_dict[item[2]] for item in data]

        return SNLI(x1, x2, y) # package

    def _data_process(self, data_file, data_size):
        data = self._data_read(data_file, data_size)
        self._vocab_update(data)
        return self._data_vectorization(data)

    def collate_fn(self, batch_data):
        sents1, sents2, relations = zip(*batch_data)
        sents1 = [torch.LongTensor(sent) for sent in sents1]
        sents2 = [torch.LongTensor(sent) for sent in sents2]
        padded_sents1 = pad_sequence(sents1, batch_first=True, padding_value=self.PAD)
        padded_sents2 = pad_sequence(sents2, batch_first=True, padding_value=self.PAD)

        return torch.LongTensor(padded_sents1), torch.LongTensor(padded_sents2), torch.LongTensor(relations)