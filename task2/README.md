# Text Classification based on Deep learning

Represent text with pretrained word embedding first, then realize text classification with neural networks (CNN and RNN). Finally analysis the performance of different word embedding and different neural networks through experiments. See more details in [report](./report.md) (written in Chinese).

Here's the raw description from [nlp-beginner](https://github.com/FudanNLP/nlp-beginner):

熟悉Pytorch，用Pytorch重写《任务一》，实现CNN、RNN的文本分类；

1. 参考
   1. https://pytorch.org/
   2. Convolutional Neural Networks for Sentence Classification https://arxiv.org/abs/1408.5882
   3. https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
2. word embedding 的方式初始化
3. 随机embedding的初始化方式
4. 用glove 预训练的embedding进行初始化 https://nlp.stanford.edu/projects/glove/
5. 知识点：
   1. CNN/RNN的特征抽取
   2. 词嵌入
   3. Dropout
6. 时间：两周

## Usage

1. Download needed pretrained word embedding (see [pretrained_emb/README.md](./pretrained_emb/README.md))
2. Run main.py
3. Check results in out directory