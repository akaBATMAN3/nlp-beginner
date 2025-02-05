# Language Model based on Neural Network

Generate Chinese poetry with LSTM and GRU. More details in [report](./report.md).

Here's the raw description from [nlp-beginner](https://github.com/FudanNLP/nlp-beginner):

用LSTM、GRU来训练字符级的语言模型，计算困惑度

1. 参考
   1. 《[神经网络与深度学习](https://nndl.github.io/)》 第6、15章
2. 数据集：poetryFromTang.txt
3. 实现要求：Pytorch
4. 知识点：
   1. 语言模型：困惑度等
   2. 文本生成
5. 时间：两周

## Usage

1. Run main.py to train a model, results are saved in out directory
2. Run inference.py to generate poetry with trained model