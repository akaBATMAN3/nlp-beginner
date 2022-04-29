import torch.nn as nn
import torch


class WordEmbedding:
    def __init__(
        self,
        word_emb_type,
        pretrained_emb_file,
        word2index_dict,
        word_emb_size
    ):
        if word_emb_type == "random":
            weight = nn.init.xavier_normal_(torch.Tensor(len(word2index_dict), word_emb_size))
        else:
            raise Exception("Unknown word embedding type!")
        self.word_emb_layer = nn.Embedding(
            num_embeddings=len(word2index_dict),
            embedding_dim=word_emb_size,
            _weight=weight
        )

class SumModel(nn.Module):
    def __init__(
        self,
        word2index_dict,
        index2word_dict,
        pretrained_emb_file,
        word_emb_size,
        input_size,
        hidden_size,
        word_emb_type="random",
        dropout=0.5,
        rnn_type="lstm",
        num_layers=1
    ):
        super(SumModel, self).__init__()
        self.word2index_dict = word2index_dict
        self.index2word_dict = index2word_dict
        self.PAD = word2index_dict["<pad>"]
        self.START = word2index_dict["<start>"]
        self.END = word2index_dict["<end>"]

        self.hidden_size = hidden_size
        self.rnn_type=rnn_type

        self.word_emb_layer = WordEmbedding(
            word_emb_type,
            pretrained_emb_file,
            word2index_dict,
            word_emb_size
        ).word_emb_layer
        self.dropout = nn.Dropout(dropout)
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
        elif rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
        else:
            raise Exception("Unknown rnn type!")
        self.linear = nn.Linear(hidden_size, len(word2index_dict))

    def forward(self, x):
        x = self.word_emb_layer(x) # [batch size, poem len, word emb size]
        x = self.dropout(x)
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x) # [batch size, poem len, hidden size]
        prob = self.linear(x) # [batch size, poem len, total word len]

        return prob

    def generate(self, device, heads=None, poems_size=4, max_poem_len=30):
        if heads is not None:
            poems_size = len(heads)
            for head in heads:
                if head not in self.word2index_dict:
                    raise Exception("Head:" + head + " is not included in the word dictionary, try another!")

        poems = list()
        for i in range(poems_size):
            if heads is not None:
                word = torch.LongTensor([self.word2index_dict[heads[i]]]).to(device)
                poem = [heads[i]]
            else:
                word = torch.LongTensor([self.START]).to(device)
                poem = list()
            h_n = torch.randn(1, 1, self.hidden_size).to(device)
            c_n = torch.randn(1, 1, self.hidden_size).to(device)
            while len(poem) < max_poem_len:
                word = torch.LongTensor([word]).to(device)
                word = self.word_emb_layer(word)
                word = self.dropout(word).view(1, 1, -1)
                if self.rnn_type == "lstm":
                    output, (h_n, c_n) = self.rnn(word, (h_n, c_n))
                elif self.rnn_type == "gru":
                    output, h_n = self.rnn(word, h_n)
                output = self.linear(output)
                word = output.topk(1)[1][0].item()

                poem.append(self.index2word_dict[word])
                if word == self.START or word == self.PAD: # invalid
                    i -= 1
                    break
                elif word == self.word2index_dict["。"]:
                    break
                elif word == self.END:
                    poem[-1] = "。"
                    break
            if poem:
                poems.append(poem)
            else:
                i -= 1

        return poems