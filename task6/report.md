# Sentence Similarity Calculation

## 1	前言

​	文本语义相似度是许多NLP应用的基础，如文本分类。以单词为例，我们常常将单词表示为词向量形式，然后采用余弦相似度等方法计算两个词向量，即两个单词间的相似度。而实际中往往需要考虑句子级的相似度计算，为此我们首先需要将句子用合适的方法表示，然后再计算其相似度。在本项目中，采用的数据集为750行成对的英文句子，首先利用统计模型、词嵌入模型将句子表示为合适的向量形式，然后用余弦相似度、词移距离来计算句子向量间的相似度，最后用Pearson相关系数对我们的计算方法进行评估。

## 2	实现

​	实现过程一共分为三步：对句子进行编码，对编码后的句子对分别计算其相似度，以及对我们的计算结果进行评估。

### 2.1	句子编码

#### 2.1.1	统计模型

​	本项目采用三种统计模型对句子编码：Bag of words，N-Gram，TF-IDF。

1. Bag of Words

   ​	所有句子向量的长度均为预先建好的词表大小，将句子中的单词根据其在词表中的位置映射到对应句子向量中的位置，具体表现为对应句子向量中的位置每次+1。缺点是没有考虑语序。

2. N-Gram

   ​	在BoW（Bag of Words）的基础上考虑了一定语序关系，进一步分为unigram（仅考虑单个单词），bigram（考虑两两相的单词），trigram（连续的三个单词）。

3. TF-IDF

   ​	采用每个单词对于其所在句子的TF-IDF值来表示句子向量（句子向量长度均为词表大小），计算方法见下公式。可以注意到，当TF中的分子变大时，说明该单词有着较强的区分度，而当IDF中分母变大时则说明该单词的区分度较低。
   $$
   TF=\frac{当前单词在其所在句子中出现次数}{当前单词所在句子的单词数}\\
   IDF=log(\frac{总句子数}{包含当前单词的句子数})\\
   当前单词对于其所在句子的TF-IDF值=TF*IDF
   $$
   注：这里关于TF-IDF的描述根据数据集作了自适应，标准解释见[tf-idf - Wikipedia](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

#### 2.1.2	词嵌入模型

​	词嵌入模型指利用预训练好的词向量来表示句子向量，最简单的方法是将一句句子中的所有词向量求和平均作为该该句子的句子向量；还可以将2.1.1中的统计模型TF-IDF加入作为权重，对词向量进行加权求和再取平均；另外，还可以采用预训练句子编码器直接对句子编码，再计算相似度，这里使用了Facebook的InferSent作为预训练的句子编码器。

​	InferSent目前有两个版本，v1/2采用glove/fastText作为预训练词向量，然后在SNLI数据集上进行训练，以下为训练过程（左图）和句子编码器结构（右图），句子编码器结构很简单，BiLSTM+max-pooling。

<img src="https://raw.githubusercontent.com/akaBATMAN3/nlp-beginner/master/imgs/202205111804098.png" alt="image-20220509204829452" style="zoom: 67%;" /><img src="https://raw.githubusercontent.com/akaBATMAN3/nlp-beginner/master/imgs/202205111806383.png" alt="image-20220509205347757" style="zoom:67%;" />

### 2.2	相似度计算

1. 余弦相似度

   计算两个句子向量间的夹角余弦值来评估其相似度，对于向量$a$和$b$的余弦相似度计算公式如下。
   $$
   \frac{a·b}{|a||b|}=\frac{\sum_{i=1}^na_i\times b_i}{\sqrt{\sum_i^n(a_i)^2}\times \sqrt{\sum_i^n(b_i)^2}}
   $$

2. 词移距离

   在高维语义空间（这里的高维语义空间指词向量空间）中，计算一句句子中所有单词转移到另一句子中所有单词的最短累积距离。

   <img src="https://raw.githubusercontent.com/akaBATMAN3/nlp-beginner/master/imgs/202205111814684.png" alt="微信图片_20220511181350" style="zoom: 50%;" />

### 2.3	评估

​	采用皮尔逊相关系数来衡量我们的计算结果（750个相似度分数）和真值之间的线性相关性，计算公式如下。相关系数的绝对值越大，相关性越强。
$$
r=\frac{\sum_{i=1}^n(x_i-\overline{x})(y_i-\overline{y})}{\sqrt{\sum_{i=1}^n(x_i-\overline{x})^2}\sqrt{\sum_{i=1}^n(y_i-\overline{y})^2}}
$$

## 3	实验

​	在实验中不仅对上述方法的不同组合进行测试，还测试了去除停止词后的表现、在不同预训练词向量上的表现，见下图。

<img src="https://raw.githubusercontent.com/akaBATMAN3/nlp-beginner/master/imgs/202205111814621.png" alt="comparison" style="zoom: 80%;" />

- 在第0-5组实验中，发现三种统计模型中TF-IDF表现最优（并且全局最优），还可以发现去除停止词后表现有提升。
- 在第6-11组实验中，发现词嵌入模型表现整体不如统计模型，还可以发现预训练词向量word2vec和glove间差异并不明显。在第12组实验中发现预训练句子编码器表现也一般。

​	总结一下，采用TF-IDF的统计模型并去除停止词可以达到最优；统计模型略优于词嵌入模型，由于采用的数据量较小，因此可能无法完全发挥词嵌入模型的作用。

## 参考

[1] [Comparing Sentence Similarity Methods](http://nlp.town/blog/sentence-similarity/)

[2] https://github.com/facebookresearch/InferSent
