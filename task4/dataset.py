import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class CONLL(Dataset):
    def __init__(self, sents, sents_tags):
        self.sents = sents
        self.sents_tags = sents_tags

    def __getitem__(self, item):
        return self.sents[item], self.sents_tags[item]

    def __len__(self):
        return len(self.sents_tags)

class CONLLDataset:

    word_pad_value = 0
    word_pad_token = "#pad#"

    def __init__(self, data_file, data_size, tag_dict):
        self.tag_dict = tag_dict
        self.vocab = dict()
        self.vocab[self.word_pad_token] = self.word_pad_value
        self.max_sent_len = 0

        self.train = self._data_process(data_file["train"], data_size["train"])
        self.val = self._data_process(data_file["val"], data_size["val"])
        self.test = self._data_process(data_file["test"], data_size["test"])

    def _data_read(self, data_file, data_size):
        with open(data_file) as f:
            data = f.readlines()
        data = data[2:] # remove header

        sents = list()
        sents_tags = list()
        sent = list()
        sent_tags = list()
        for line in data:
            if line != '\n':
                item = line.split()
                if item[0] == "-DOCSTART-":
                    continue
                sent.append(item[0].lower()) # collect words to form sentence
                sent_tags.append(item[-1])
            else:
                if sent and sent_tags:
                    sents.append(sent) # collect sentences
                    sents_tags.append(sent_tags)
                sent = list() # can't use list.clear(), which will clear sents at the same time
                sent_tags = list()
            if len(sents) == data_size:
                break
        if sent and sent_tags:
            sents.append(sent)
            sents_tags.append(sent_tags)
        assert len(sents) == len(sents_tags)

        return sents, sents_tags

    def _data_sort(self, sents, sents_tags):
        """Sort the sentences from short to long along with their tags."""
        data_zip = list(zip(sents, sents_tags))
        data_zip.sort(key=lambda item: len(item[0])) # len(item[0]) is the length of one sentence
        return zip(*data_zip) # unzip the sorted data

    def _dict_update(self, sents, sents_tags):
        """Update the vocabulary and tag dictionary."""
        for sent in sents:
            for word in sent:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
        for sent_tags in sents_tags:
            for tag in sent_tags:
                if tag not in self.tag_dict:
                    self.tag_dict[tag] = len(self.tag_dict)

    def _data_vectorization(self, sents, sents_tags):
        sents_matrix = list()
        sents_tags_matrix = list()
        for sent in sents:
            word_indices = [self.vocab[word] for word in sent]
            self.max_sent_len = max(self.max_sent_len, len(word_indices))
            sents_matrix.append(word_indices)
        for sent_tags in sents_tags:
            tag_indices = [self.tag_dict[tag] for tag in sent_tags]
            sents_tags_matrix.append(tag_indices)

        return CONLL(sents_matrix, sents_tags_matrix)

    def _data_process(self, data_file, data_size):
        sents, sents_tags = self._data_read(data_file, data_size)
        sents, sents_tags = self._data_sort(sents, sents_tags)
        self._dict_update(sents, sents_tags)
        return self._data_vectorization(sents, sents_tags)

    def collate_fn(self, batch_data):
        sents, sents_tags = zip(*batch_data)
        sents = [torch.LongTensor(sent) for sent in sents]
        sents_tags = [torch.LongTensor(sent_tags) for sent_tags in sents_tags]
        padded_sents = pad_sequence(sents, batch_first=True, padding_value=self.word_pad_value)
        padded_sents_tags = pad_sequence(sents_tags, batch_first=True, padding_value=self.tag_dict["<pad>"])

        return torch.LongTensor(padded_sents), torch.LongTensor(padded_sents_tags)