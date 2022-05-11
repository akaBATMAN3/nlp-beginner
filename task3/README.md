# Natural Language Inference based on Attention Mechanism

Judge the relationship between sentence pairs with neural network based on attention mechanism. See more details in [report](./report.md).

Here's the raw description from [nlp-beginner](https://github.com/FudanNLP/nlp-beginner):

输入两个句子判断，判断它们之间的关系。参考[ESIM](https://arxiv.org/pdf/1609.06038v3.pdf)（可以只用LSTM，忽略Tree-LSTM），用双向的注意力机制实现。

1. 参考
   1. 《[神经网络与深度学习](https://nndl.github.io/)》 第7章
   2. Reasoning about Entailment with Neural Attention https://arxiv.org/pdf/1509.06664v1.pdf
   3. Enhanced LSTM for Natural Language Inference https://arxiv.org/pdf/1609.06038v3.pdf
2. 数据集：https://nlp.stanford.edu/projects/snli/
3. 实现要求：Pytorch
4. 知识点：
   1. 注意力机制
   2. token2token attetnion
5. 时间：两周

## Usage

1. Download needed data from [data/README.md](./data/README.md)
2. Run main.py
3. Check results in out directory