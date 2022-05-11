# Text Classification based on Machine Learning

## 1	文本特征表示方法

### 1.1	Bag of words

​	类似one-hot编码。每段文本中，某个单词若出现，则根据词表将对应位置下标设为1。每个单词的向量两两正交，没有相关性。

### 1.2	N-grams

​	Bag of words进阶版，除了考虑单个单词，还考虑两两相邻（2-gram）的单词，体现为：word1_word2。

## 2	Softmax的向量表示

$$
\widehat{y}=softmax(W^Tx)=\frac{exp(W^Tx)}{1_C^Texp(W^Tx)}
$$

​	训练过程中用训练数据集来计算（拟合）上式中的W，使其能将训练数据正确分类，之后将W与测试集运算看其是否能将测试集正确分类。

## 3	梯度下降

### 3.1	SGD

$$
ΔW=-x^{(n)}(y^{(n)}-\widehat{y}_W^{(n)})^T \\
W_{t+1}←W_t-\alpha(-ΔW_t)
$$

​	每次循环只抽取一个训练样本计算梯度，若训练样本大小为1000，循环次数为1000，则正好针对所有训练样本计算了一次梯度。

### 3.2	BGD

$$
ΔW=-\frac{1}{N}\sum_{n=1}^N x^{(n)}(y^{(n)}-\widehat{y}_W^{(n)})^T \\
W_{t+1}←W_t-\alpha(-ΔW_t) \\
其中N为全部样本数量
$$

​	每次循环针对所有训练样本计算一次梯度，若训练样本大小为1000，循环次数为1000，则针对所有训练样本计算了1000次梯度。

### 3.3	mini-batch

​	BGD和SGD的折中方案，每次循环针对一个mini_size计算梯度，若训练样本大小为1000，循环次数为1000，mini_size=10，则针对所有训练样本计算了mini_size*循环次数/训练样本大小=10次梯度。
$$
ΔW=-\frac{1}{K}\sum_{n\in S} x^{(n)}(y^{(n)}-\widehat{y}_W^{(n)})^T \\
W_{t+1}←W_t-\alpha(-ΔW_t) \\
其中K为小批量样本数量
$$

## 4	实验结果

​	不同学习率下不同方法在训练集和测试集上的准确率。

<img src="https://raw.githubusercontent.com/akaBATMAN3/nlp-beginner/master/imgs/202205111705165.png" alt="10000epochs_comparison" style="zoom: 80%;" />