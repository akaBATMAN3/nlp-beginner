# Download and Preprocess

Download [glove.840B.300d.zip](https://nlp.stanford.edu/projects/glove/) and [word2vec](https://github.com/mmihaltz/word2vec-GoogleNews-vectors) and unzip to this directory. Besides, to load Glove, you have to run the following code to convert the downloaded Glove file to word2vec format and then load the embeddings into a Gensim model:

```python
from gensim.scripts.glove2word2vec import glove2word2vec
glove2word2vec("task6/pretrained_emb/glove.840B.300d.txt", "task6/pretrained_emb/glove.840B.300d.w2v.txt")
```

