import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WordEmbedding:
    def __init__(
        self,
        word_emb_type,
        pretrained_emb_file,
        vocab,
        word_emb_size
    ):
        if word_emb_type == "glove":
            weight = self._pretrained_weight(pretrained_emb_file, vocab, word_emb_size)
        else:
            weight = nn.init.xavier_normal_(torch.Tensor(len(vocab), word_emb_size))
        assert len(vocab) == len(weight)
        self.word_emb_layer = nn.Embedding(
            num_embeddings=len(vocab),
            embedding_dim=word_emb_size,
            _weight=weight
        )

    def _pretrained_weight(self, pretrained_emb_file, vocab, word_emb_size):
        with open(pretrained_emb_file, 'rb') as f:
            lines = f.readlines()
        pretrained_emb_dict = dict() # {word:word embedding}
        for i in range(len(lines)):
            line = lines[i].split()
            pretrained_emb_dict[line[0].decode("utf-8").lower()] = [float(line[j]) for j in range(1, 51)]

        emb_matrix = list()
        for word in vocab:
            if word in pretrained_emb_dict:
                emb_matrix.append(pretrained_emb_dict[word])
            else:
                emb_matrix.append(np.zeros(word_emb_size))

        return torch.tensor(np.array(emb_matrix), dtype=torch.float)

class Text_RNN(nn.Module):
    def __init__(
        self,
        word_emb_type,
        pretrained_emb_file,
        vocab,
        word_emb_size,
        input_size,
        hidden_size,
        category_size,
        dropout=0.5,
        num_layers=1,
        nonlinearity='tanh',
    ):
        super(Text_RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.word_emb_layer = WordEmbedding(
            word_emb_type,
            pretrained_emb_file,
            vocab,
            word_emb_size
        ).word_emb_layer
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, category_size)

    def forward(self, x):
        if x.device == torch.device("cpu"):
            x = torch.LongTensor(x) # [batch size, sent len]
        else:
            x = torch.cuda.LongTensor(x)
        batch_size = x.size(0)
        x = self.word_emb_layer(x) # [batch size, sent len, word emb size]
        x = self.dropout(x)

        h_0 = torch.autograd.Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(x.device)
        _, h_n = self.rnn(x, h_0) # [1, batch size, hidden size]
        prob = self.linear(h_n).squeeze(0) # [batch size, categoiry size]

        return prob

class Text_CNN(nn.Module):
    def __init__(
        self,
        word_emb_type,
        pretrained_emb_file,
        vocab,
        word_emb_size,
        out_channels,
        category_size,
        dropout=0.5,
    ):
        super(Text_CNN, self).__init__()
        self.word_emb_size = word_emb_size
    
        self.word_emb_layer = WordEmbedding(
            word_emb_type,
            pretrained_emb_file,
            vocab,
            word_emb_size
        ).word_emb_layer
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, out_channels, (2, word_emb_size), padding=(1, 0)), # kernel size: 2*word_emb_size
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, out_channels, (3, word_emb_size), padding=(1, 0)), # pad 1*50 to 3*50 in case of there's only one word in sentences, then kernel works
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1, out_channels, (4, word_emb_size), padding=(2, 0)),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(1, out_channels, (5, word_emb_size), padding=(2, 0)),
            nn.ReLU()
        )
        self.linear = nn.Linear(4 * out_channels, category_size)

    def forward(self, x):
        if x.device == torch.device("cpu"):
            x = torch.LongTensor(x)
        else:
            x = torch.cuda.LongTensor(x)
        x = self.word_emb_layer(x).view(x.shape[0], 1, x.shape[1], self.word_emb_size) # [batch size, 1, sent len, word emb size]
        x = self.dropout(x)

        conv1 = self.conv1(x).squeeze(3) # [batch size, out channels, sent len + 2 - 1]
        pool1 = F.max_pool1d(conv1, conv1.shape[2]) # [batch size, out channels, 1]
        conv2 = self.conv2(x).squeeze(3) # [batch size, out channels, sent len + 2 - 2]
        pool2 = F.max_pool1d(conv2, conv2.shape[2])
        conv3 = self.conv3(x).squeeze(3) # [batch size, out channels, sent len + 4 - 3]
        pool3 = F.max_pool1d(conv3, conv3.shape[2])
        conv4 = self.conv4(x).squeeze(3) # [batch size, out channels, sent len + 4 - 4]
        pool4 = F.max_pool1d(conv4, conv4.shape[2])
        pool = torch.cat([pool1, pool2, pool3, pool4], 1).squeeze(2) # [batch size, 4*out channels]
        prob = self.linear(pool) # [batch size, category_size]

        return prob