# Text Classification based on Machine Learning

Represent text with Bag-of-words and N-Gram first, then realize text classification with softmax regression. Finally analysis the performance of different text representation methods, different gradient strategies and different learning rates through experiments. See more details in [report](./report.md) (written in Chinese).

Here's the raw description from [nlp-beginner](https://github.com/FudanNLP/nlp-beginner):

实现基于logistic/softmax regression的文本分类

1. 参考
   1. [文本分类](https://github.com/FudanNLP/nlp-beginner/blob/master/文本分类.md)
   2. 《[神经网络与深度学习](https://nndl.github.io/)》 第2/3章
2. 数据集：[Classify the sentiment of sentences from the Rotten Tomatoes dataset](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews)
3. 实现要求：NumPy
4. 需要了解的知识点：
   1. 文本特征表示：Bag-of-Word，N-gram
   2. 分类器：logistic/softmax regression，损失函数、（随机）梯度下降、特征选择
   3. 数据集：训练集/验证集/测试集的划分
5. 实验：
   1. 分析不同的特征、损失函数、学习率对最终分类性能的影响
   2. shuffle 、batch、mini-batch
6. 时间：两周

## Usage

1. Download needed data (see [data/README.md](./data/README.md))
2. Run main.py
3. Check results in out directory