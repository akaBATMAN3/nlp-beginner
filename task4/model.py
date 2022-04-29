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
        self.word_emb_layer = nn.Embedding(num_embeddings=len(vocab), embedding_dim=word_emb_size, _weight=weight)

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

class CRF(nn.Module):
    def __init__(self, tag_dict):
        super(CRF, self).__init__()
        self.PAD = tag_dict["<pad>"]
        self.START = tag_dict["<start>"]
        self.END = tag_dict["<end>"]

        A = torch.zeros(len(tag_dict), len(tag_dict))
        A[:, self.START] = -10000.0
        A[self.END, :] = -10000.0
        A[:, self.PAD] = -10000.0
        A[self.PAD, :]= -10000.0
        A[self.PAD, self.PAD] = 0.0 # <pad> to <pad> is allowed
        A[self.PAD, self.END] = 0.0 # <pad> to <end> is allowed
        self.A = nn.Parameter(A) # A_{i,j} means the transfer score of tag i to tag j

    def _loss_calc(self, P, y, mask):
        r"""
        Calculate loss using dp.
        loss = -log(exp(true_score) / total_score) = -(true_score - log(total_score))

        .. math::
            loss=-log\ p(y|X)\\
            p(y|X)=\frac{exp(s(X,y))}{\sum_{\tilde{y}\in Y_X}exp(s(X,\tilde{y}))}\\
            s(X,y)=\sum_{i=0}^nA_{y_i,y_{i+1}}+\sum_{i=1}^nP_{i,y_i}\\
            where :math:`Y_X` represents all possible tag sequences, math:`A_{i,j}` means the transfer score of tag i to tag j,
            and math:`P_{i,j}` means the score of math:`j^{th}` tag of math:`i^{th}` word.
        """
        batch_size, sent_len, tag_num = P.shape

        true_score = torch.zeros(batch_size).to(P.device)
        for i in range(sent_len):
            if i == 0:
                first_tag = y[:, 0] # first tag of all sents, [batch size]
                tag2tag_score = self.A[self.START, first_tag]
                word2tag_score = torch.gather(P[:, 0], 1, first_tag.unsqueeze(1)).squeeze(1)
                true_score += tag2tag_score + word2tag_score
            else:
                non_pad = mask[:, i] # whether current tag is non-pad tag
                pre_tag = y[:, i - 1]
                cur_tag = y[:, i]
                tag2tag_score = self.A[pre_tag, cur_tag]
                word2tag_score = torch.gather(P[:, i], 1, cur_tag.unsqueeze(1)).squeeze(1)
                true_score += (tag2tag_score + word2tag_score) * non_pad
        last_tag_index = mask.sum(1) - 1
        last_tag = torch.gather(y, 1, last_tag_index.unsqueeze(1)).squeeze(1) # last non-pad tag of all sents
        true_score += self.A[last_tag, self.END]

        total_score = self.A[self.START, :].unsqueeze(0) + P[:, 0] # score of <start> to all tags + score of the first word's all tags
        for i in range(1, sent_len):
            tag_total_score = list() # each tag's current total score
            for j in range(tag_num):
                tag2tag_score = self.A[:, j].unsqueeze(0) # score of all tags to current tag
                word2tag_score = P[:, i, j].unsqueeze(1) # score of current word's current tag
                tag_total_score.append(torch.logsumexp(total_score + tag2tag_score + word2tag_score, dim=1))
            new_total_score = torch.stack(tag_total_score).t()
            non_pad = mask[:, i].unsqueeze(-1)
            total_score = new_total_score * non_pad + total_score * (1 - non_pad)
        total_score += self.A[:, self.END].unsqueeze(0)

        return -torch.sum(true_score - torch.logsumexp(total_score, dim=1))

    def _predict(self, P, mask):
        r"""
        Compute the maximum a posteriori sequence using viterbi algorithm.

        .. math::
            y^*=\underset{\tilde{y}\in Y_X}{argmax}\ s(X,\tilde{y})
        """
        batch_size, sent_len, tag_num = P.shape
        
        prob = self.A[self.START, :].unsqueeze(0) + P[:, 0]
        tag = torch.cat([torch.tensor(range(tag_num)).view(1, -1, 1) for _ in range(batch_size)], dim=0).to(P.device)
        for i in range(1, sent_len):
            new_prob = torch.zeros(batch_size, tag_num).to(P.device)
            new_tag = torch.zeros(batch_size, tag_num, 1).to(P.device)
            for j in range(tag_num):
                temp_prob = prob + self.A[:, j].unsqueeze(0) + P[:, i, j].unsqueeze(1) # compute each tag's current score
                max_prob, max_tag = torch.max(temp_prob, dim=1) # collect max score and it's corresponding index
                new_prob[:, j] = max_prob
                new_tag[:, j, 0] = max_tag

            non_pad = mask[:, i].unsqueeze(-1)
            prob = new_prob * non_pad + prob * (1 - non_pad)

            non_pad = non_pad.unsqueeze(-1)
            temp_tag = torch.cat([torch.tensor(range(tag_num)).view(1, -1, 1) for _ in range(batch_size)], dim=0).to(P.device)
            append_tag = temp_tag * non_pad + torch.ones(batch_size, tag_num, 1).to(P.device) * self.PAD * (1 - non_pad)
            new_tag = new_tag.long()
            pre_tag = tag[[[i] * tag_num for i in range(batch_size)], new_tag[:, :, 0], :]
            tag = torch.cat([pre_tag, append_tag], dim=-1)
        prob += self.A[:, self.END].unsqueeze(0)
        max_tag = prob.argmax(-1)

        return tag[[i for i in range(batch_size)], max_tag]

    def forward(self, P, y, mask):
        """See more details in https://towardsdatascience.com/implementing-a-linear-chain-conditional-random-field-crf-in-pytorch-16b0b9c4b4ea"""
        loss = self._loss_calc(P, y, mask)
        pred = self._predict(P, mask)
        return loss, pred

class LSTM_CRF(nn.Module):
    def __init__(
        self,
        pretrained_emb_file,
        vocab,
        word_emb_size,
        input_size,
        hidden_size,
        tag_dict,
        word_emb_type="random",
        dropout=0.5,
        num_layers=1,
        bidirectional=True,
    ):
        super(LSTM_CRF, self).__init__()
        self.word_emb_layer = WordEmbedding(
            word_emb_type,
            pretrained_emb_file,
            vocab,
            word_emb_size
        ).word_emb_layer
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.linear = nn.Linear(2*hidden_size, len(tag_dict))
        self.crf = CRF(tag_dict)

    def forward(self, x, y, mask):
        x = self.word_emb_layer(x)
        x = self.dropout(x)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x) # [batch size, sent len, 2*hidden size]
        P = self.linear(x) # P_{i,j} means the score of j^{th} tag of i^{th} word, [batch size, sent len, tag num]
        loss, pred = self.crf(P, y, mask)

        return loss, pred