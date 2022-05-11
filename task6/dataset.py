import pandas as pd
import re
import nltk
import numpy as np
import collections
from nltk.stem import WordNetLemmatizer
from gensim.models.keyedvectors import KeyedVectors


class STSDateset:
    def __init__(
        self,
        data_file,
        data_size,
        stopwords,
        method,
        word_emb_type,
        pretrained_emb_file,
        gram_num=3,
        ngram_threshold=2,
        word_emb_size=300
    ):
        self.stopwords = nltk.corpus.stopwords.words('english') if stopwords else None
        self.method = method
        self.gram_num = gram_num
        self.ngram_threshold = ngram_threshold
        self.vocab = dict() # {word:word's index}
        self.sent_freq = dict() # {word:freq of sents including the word}
        if method in ["avg", "avg-tfidf", "wmd"]:
            if word_emb_type == "word2vec":
                self.word_emb = KeyedVectors.load_word2vec_format(pretrained_emb_file["word2vec"], binary=True)
            elif word_emb_type == "glove":
                self.word_emb = KeyedVectors.load_word2vec_format(pretrained_emb_file["glove"])
            else:
                raise Exception("Unknown word embedding type!")
        self.word_emb_size = word_emb_size

        self.x, self.y = self._data_process(data_file, data_size)

    def _data_clean(self, sents):
        """Word segmentation, lemmatization and stopwords removal."""
        sents = sents.apply((lambda x: re.sub('[^a-zA-Z]', ' ', x))) # word segmentation
        lemmatizer = WordNetLemmatizer()
        clean_sents = list()
        for sent in sents:
            words = sent.lower().split()
            words = [lemmatizer.lemmatize(word) for word in words] # lemmatization
            if self.stopwords is not None: # stopwords removal
                clean_sent = list()
                for word in words:
                    if word not in self.stopwords:
                        clean_sent.append(word)
            else:
                clean_sent = words
            clean_sents.append(clean_sent)

        return clean_sents

    def _dict_update(self, sents):
        """Update word vocabulary and dict of frequences of sentences(tfidf)."""
        for sent in sents: # build vocabulary with single words
            for word in sent:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
        if self.method == "ngram": # extend vocab with ngrams
            ngram_freq = dict()
            for n in range(2, self.gram_num + 1):
                for sent in sents:
                    for i in range(len(sent) - n + 1):
                        word = '_'.join(sent[i:i + n])                     
                        if word not in ngram_freq:
                            ngram_freq[word] = 1
                        else:
                            ngram_freq[word] += 1
            for word, freq in ngram_freq.items(): # remove low freq ngrams
                if freq >= self.ngram_threshold:
                    self.vocab[word] = len(self.vocab)
        elif self.method in ["tfidf", "avg-tfidf"]:
            for word in self.vocab:
                self.sent_freq[word] = 0
                for sent in sents:
                    if word in sent:
                        self.sent_freq[word] += 1

    def _data_vectorization(self, sents):
        """Convert sents into sent vectors."""
        if self.method in ["bow", "ngram", "tfidf"]: # init matrix
            sents_matrix = np.zeros((len(sents), len(self.vocab)))
        elif self.method in ["avg", "avg-tfidf"]:
            sents_matrix = np.zeros((len(sents), self.word_emb_size))

        if self.method in ["bow", "ngram"]:
            for i in range(len(sents)):
                for word in sents[i]:
                    sents_matrix[i][self.vocab[word]] += 1
        if self.method == "ngram":
            for n in range(2, self.gram_num + 1):
                for i in range(len(sents)):
                    for j in range(len(sents[i]) - n + 1):
                        word = '_'.join(sents[i][j:j + n])
                        if word in self.vocab:
                            sents_matrix[i][self.vocab[word]] += 1
        elif self.method == "tfidf":
            for i in range(len(sents)):
                word_freq = collections.Counter(sents[i])
                for word in word_freq.keys():
                    TF = word_freq[word] / len(sents[i]) # freq of word in sent / num of words in sent
                    IDF = np.log(len(sents) * 2 /(self.sent_freq[word] + 1)) # log(total num of sents / (num of sents including this word + 1))
                    sents_matrix[i][self.vocab[word]] = TF * IDF
        elif self.method in "avg":
            for i in range(len(sents)):
                sent_emb = np.zeros((len(sents[i]), self.word_emb_size))
                for j, word in enumerate(sents[i]):
                    if word in self.word_emb:
                        sent_emb[j] = self.word_emb[word]
                    else:
                        sent_emb[j] = self.word_emb["unk"]
                sents_matrix[i] = np.mean(sent_emb, axis = 0)
        elif self.method == "avg-tfidf":
            for i in range(len(sents)):
                word_freq = collections.Counter(sents[i])
                weights = [word_freq[word] / len(sents[i]) * np.log(len(sents) / (self.sent_freq[word] + 1))
                           for word in word_freq.keys()]
                sent_emb = np.zeros((len(word_freq), self.word_emb_size))
                for j, word in enumerate(word_freq):
                    if word in self.word_emb:
                        sent_emb[j] = self.word_emb[word]
                    else:
                        sent_emb[j] = self.word_emb["unk"]
                sents_matrix[i] = np.average(sent_emb, axis=0, weights=weights)

        return sents_matrix

    def _data_process(self, data_file, data_size):
        data = pd.read_csv(data_file, sep='\t') if data_size is None else pd.read_csv(data_file, sep='\t', nrows=data_size)
        sents1 = self._data_clean(data["Sen1"])
        sents2 = self._data_clean(data["Sen2"])
        if self.method not in ["avg", "wmd", "infersent"]:
            self._dict_update(sents1 + sents2)
        if self.method not in ["wmd", "infersent"]:
            sents1 = self._data_vectorization(sents1)
            sents2 = self._data_vectorization(sents2)
        if self.method is "infersent": # reorganize discrete words into coherent sents
            sents1 = [' '.join(sent) for sent in sents1]
            sents2 = [' '.join(sent) for sent in sents2]

        return sents1, sents2