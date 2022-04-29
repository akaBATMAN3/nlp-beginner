import torch
from torch.nn.utils.rnn import pad_sequence


class Poem:
    def __init__(self, poems):
        self.poems = poems

    def __getitem__(self, item):
        return self.poems[item]

    def __len__(self):
        return len(self.poems)

class PoemDataset:
    def __init__(self, data_file, data_size):
        self.word2index_dict = {"<pad>": 0, "<start>": 1, "<end>": 2}
        self.index2word_dict = {0: "<pad>", 1: "<start>", 2: "<end>"}
        self.train = self._data_process(data_file["train"], data_size["train"])

    def _data_read(self, data_file, data_size):
        with open(data_file, encoding='utf8') as f:
            items = f.readlines()

        poems = list()
        poem = ""
        for item in items:
            if item != '\n':
                poem = ''.join([poem, item[: -2]]) # exclude the extra '\n' to form a whole poem
            else :
                if poem:
                    poems.append(poem)
                poem = ""
            if len(poem) == data_size:
                break

        return poems

    def _dict_update(self, poems):
        for poem in poems:
            for word in poem:
                if word not in self.word2index_dict:
                    self.word2index_dict[word] = len(self.word2index_dict)
                    self.index2word_dict[len(self.index2word_dict)] = word

    def _data_vectorization(self, poems):
        poems_matrix = list()
        for poem in poems:
            poems_matrix.append([self.word2index_dict[word] for word in poem])
        return Poem(poems_matrix)

    def _data_process(self, data_file, data_size):
        poems = self._data_read(data_file, data_size)
        poems.sort(key=lambda x: len(x)) # sort from short poem to long
        self._dict_update(poems)
        return self._data_vectorization(poems)

    def collate_fn(self, batch_data):
        poems = batch_data
        poems = [torch.LongTensor([self.word2index_dict["<start>"], *poem]) for poem in poems] # add <start> to each poem
        padded_poems = pad_sequence(poems, batch_first=True, padding_value=self.word2index_dict["<pad>"])
        padded_poems = [torch.cat([poem, torch.LongTensor([self.word2index_dict["<end>"]])]) for poem in padded_poems] # add <end> to each poem

        return torch.LongTensor(list(map(list, padded_poems)))