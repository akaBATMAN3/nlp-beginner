import numpy as np
import torch
import torch.nn as nn


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

class Input_Encoding(nn.Module):
    def __init__(
        self,
        word_emb_type,
        pretrained_emb_file,
        vocab,
        word_emb_size,
        input_size,
        hidden_size,
        num_layers=1,
        dropout=0.5
    ):
        super(Input_Encoding, self).__init__()
        self.word_emb_layer = WordEmbedding(
            word_emb_type=word_emb_type,
            pretrained_emb_file=pretrained_emb_file,
            vocab=vocab,
            word_emb_size=word_emb_size
        ).word_emb_layer
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        if x.device == torch.device("cpu"):
            x = torch.LongTensor(x)
        else:
            x = torch.cuda.LongTensor(x)
        x = self.word_emb_layer(x) # [batch size, batched sent len, word emb size]
        x = self.dropout(x) # [batch size, batched sent len, word emb size]
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x) # [batch size, batched sent len, 2*hidden size]

        return x

class Local_Inference_Modeling(nn.Module):
    def __init__(self):
        super(Local_Inference_Modeling, self).__init__()
        self.softmax_row = nn.Softmax(dim=1)
        self.softmax_col = nn.Softmax(dim=2)

    def forward(self, a_bar, b_bar):
        e = torch.matmul(a_bar, b_bar.transpose(1, 2)) # [batch size, sent1 len, sent2 len]
        a_tilde = self.softmax_col(e).bmm(b_bar) # [batch size, sent1 len, sent2 len]*[batch size, sent2 len, 2*hidden size]
        b_tilde = self.softmax_row(e).transpose(1, 2).bmm(a_bar) # [batch size, sent2 len, sent1 len]*[batch size, sent1 len, 2*hidden size]
        m_a = torch.cat([a_bar, a_tilde, a_bar - a_tilde, a_bar * a_tilde], dim=-1) # [batch size, sent1, 8*hidden size]
        m_b = torch.cat([b_bar, b_tilde, b_bar - b_tilde, b_bar * b_tilde], dim=-1) # [batch size, sent2, 8*hidden size]

        return m_a, m_b

class Inference_Composition(nn.Module):
    def __init__(
        self,
        concat_size,
        input_size,
        hidden_size,
        num_layers=1,
        dropout=0.5
    ):
        super(Inference_Composition, self).__init__()
        self.linear = nn.Linear(concat_size, input_size)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        x = self.linear(x) # [batch size, sent len, hidden size]
        x = self.dropout(x)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x) # [batch size, sent len, 2*hidden size]

        return x

class Prediction(nn.Module):
    def __init__(
        self,
        concat_size,
        hidden_size,
        type_num,
        dropout=0.5
    ):
        super(Prediction, self).__init__()
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(concat_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, type_num)
        )

    def forward(self, v_a, v_b):
        v_a_avg = v_a.sum(1) / v_a.shape[1] # [batch size, 2*hidden size]
        v_a_max = v_a.max(1)[0] # [batch size, 2*hidden size]
        v_b_avg = v_b.sum(1) / v_b.shape[1] # [batch size, 2*hidden size]
        v_b_max = v_b.max(1)[0] # [batch size, 2*hidden size]
        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=-1)

        return self.mlp(v)

class ESIM(nn.Module):
    def __init__(
        self,
        word_emb_type,
        pretrained_emb_file,
        vocab,
        word_emb_size,
        input_size,
        hidden_size,
        type_num,
        num_layers=1,
        dropout=0.5
    ):
        super(ESIM, self).__init__()
        self.input_encoding = Input_Encoding(
            word_emb_type=word_emb_type,
            pretrained_emb_file=pretrained_emb_file,
            vocab=vocab,
            word_emb_size=word_emb_size,
            dropout=dropout,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        self.local_inference_modelig = Local_Inference_Modeling()
        self.inference_composition = Inference_Composition(
            concat_size=hidden_size*8,
            input_size=input_size,
            dropout=dropout,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        self.prediction = Prediction(
            dropout=dropout,
            concat_size=hidden_size*8,
            hidden_size=hidden_size,
            type_num=type_num
        )

    def forward(self, a, b):
        a_bar = self.input_encoding(a) # [batch size, sent1 len, 2*hidden size]
        b_bar = self.input_encoding(b) # [batch size, sent2 len, 2*hidden size]
        m_a, m_b = self.local_inference_modelig(a_bar, b_bar) # [batch size, sent len, 8*hidden size]
        v_a = self.inference_composition(m_a) # [batch size, sent1 len, 2*hidden size]
        v_b = self.inference_composition(m_b) # [batch size, sent2 len, 2*hidden size]
        prob = self.prediction(v_a, v_b) # [batch size, type num]

        return prob