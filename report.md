# nlp-beginner



## 任务一：基于机器学习的文本分类

### 文本特征表示方法

#### Bag of words

类似one-hot编码。每段文本中，某个单词若出现，则根据词表将对应位置下标设为1。每个单词的向量两两正交，没有相关性。

#### N-grams

Bag of words进阶版，除了考虑单个单词，还考虑两两相邻（2-gram）的单词，体现为：word1_word2。



### Softmax的向量表示

$$
\widehat{y}=softmax(W^Tx)=\frac{exp(W^Tx)}{1_C^Texp(W^Tx)}
$$

训练过程中用训练数据集来计算（拟合）上式中的W，使其能将训练数据正确分类，之后将W与测试集运算看其是否能将测试集正确分类。



### 梯度下降

#### SGD

$$
ΔW=-x^{(n)}(y^{(n)}-\widehat{y}_W^{(n)})^T \\
W_{t+1}←W_t-\alpha(-ΔW_t)
$$

每次循环只抽取一个训练样本计算梯度，若训练样本大小为1000，循环次数为1000，则正好针对所有训练样本计算了一次梯度。

#### BGD

$$
ΔW=-\frac{1}{N}\sum_{n=1}^N x^{(n)}(y^{(n)}-\widehat{y}_W^{(n)})^T \\
W_{t+1}←W_t-\alpha(-ΔW_t) \\
其中N为全部样本数量
$$

每次循环针对所有训练样本计算一次梯度，若训练样本大小为1000，循环次数为1000，则针对所有训练样本计算了1000次梯度。

#### mini-batch

BGD和SGD的折中方案，每次循环针对一个mini_size计算梯度，若训练样本大小为1000，循环次数为1000，mini_size=10，则针对所有训练样本计算了mini_size*循环次数/训练样本大小=10次梯度。
$$
ΔW=-\frac{1}{K}\sum_{n\in S} x^{(n)}(y^{(n)}-\widehat{y}_W^{(n)})^T \\
W_{t+1}←W_t-\alpha(-ΔW_t) \\
其中K为小批量样本数量
$$

### 实验结果

<img src="https://raw.githubusercontent.com/akaBATMAN3/Typora_img/master/img/202204290938834.png" alt="10000epochs_comparison" style="zoom:80%;" />



## 任务二：基于深度学习的文本分类

### 文本特征表示方法

word embedding是将每个单词映射到高维空间的向量，其中不同维度可能代表不同真实世界含义。由于每个维度都由连续值来表示，因此词向量间可能产生相关性（如在每个维度上靠近彼此或一组组合有相似对立关系）。word embedding需要由训练获得，可以随机初始化或采用预训练好的word embedding。



### Text_RNN

```python
Text_RNN(
  (word_emb_layer): Embedding(16532, 50)
  (dropout): Dropout(p=0.2, inplace=False)
  (rnn): RNN(50, 50, batch_first=True)
  (linear): Linear(in_features=50, out_features=5, bias=True)
)
```

#### Dropout

在每次前向/反向传播中按照一定概率（1-p）随机drop一些神经元子集，测试阶段用全部神经元来预测



### Text_CNN

使用n*word embedding size的卷积核扫描由word embedding组成的一句句子，其中n可以设为2，3，4来挖掘词组关系。

<img src="https://raw.githubusercontent.com/akaBATMAN3/Typora_img/master/img/202204290940725.png" alt="image-20220319170235075" style="zoom:80%;" />

```python
Text_CNN(
  (word_emb_layer): Embedding(16532, 50)
  (dropout): Dropout(p=0.2, inplace=False)
  (conv1): Sequential(
    (0): Conv2d(1, 52, kernel_size=(2, 50), stride=(1, 1), padding=(1, 0))
    (1): ReLU()
  )
  (conv2): Sequential(
    (0): Conv2d(1, 52, kernel_size=(3, 50), stride=(1, 1), padding=(1, 0))
    (1): ReLU()
  )
  (conv3): Sequential(
    (0): Conv2d(1, 52, kernel_size=(4, 50), stride=(1, 1), padding=(2, 0))
    (1): ReLU()
  )
  (conv4): Sequential(
    (0): Conv2d(1, 52, kernel_size=(5, 50), stride=(1, 1), padding=(2, 0))
    (1): ReLU()
  )
  (linear): Linear(in_features=208, out_features=5, bias=True)
)
```

### 实验结果

#### 训练集

<img src="https://raw.githubusercontent.com/akaBATMAN3/Typora_img/master/img/202204290941397.png" alt="50epochs_comparison" style="zoom:80%;" />

#### 测试集

| 模型       | Acc      | Long sent acc |
| ---------- | -------- | ------------- |
| random_rnn | 57.8     | 46.3          |
| random_cnn | 56.6     | 47.1          |
| glove_rnn  | **63.8** | 49.1          |
| glove_cnn  | 59.6     | **51.5**      |



## 任务三：基于注意力机制的文本匹配

### ESIM

<img src="https://raw.githubusercontent.com/akaBATMAN3/Typora_img/master/img/202204290942679.png" alt="image-20220403155034356" style="zoom:80%;" />

```python
ESIM(
  (input_encoding): Input_Encoding(
    (word_emb_layer): Embedding(34502, 50)
    (dropout): Dropout(p=0.2, inplace=False)
    (lstm): LSTM(50, 50, batch_first=True, bidirectional=True)
  )
  (local_inference_modelig): Local_Inference_Modeling(
    (softmax_row): Softmax(dim=1)
    (softmax_col): Softmax(dim=2)
  )
  (inference_composition): Inference_Composition(
    (linear): Linear(in_features=400, out_features=50, bias=True)
    (dropout): Dropout(p=0.2, inplace=False)
    (lstm): LSTM(50, 50, batch_first=True, bidirectional=True)
  )
  (prediction): Prediction(
    (mlp): Sequential(
      (0): Dropout(p=0.2, inplace=False)
      (1): Linear(in_features=400, out_features=50, bias=True)
      (2): Tanh()
      (3): Linear(in_features=50, out_features=4, bias=True)
    )
  )
)
```

#### Input Encoding

<img src="https://raw.githubusercontent.com/akaBATMAN3/Typora_img/master/img/202204290942170.png" alt="image-20220403155822097" style="zoom:50%;" />

将句子用词嵌入表示，之后两句句子的每个单词的词嵌入经过BiLSTM提取特征，用提取的隐藏状态作为单词上下文特征

#### Local Inference Modeling

<img src="E:/typora/typora_img/image-20220403160747748.png" alt="image-20220403160747748" style="zoom:50%;" />

软对齐层计算两句句子之间的相似度作为注意力权重

<img src="https://raw.githubusercontent.com/akaBATMAN3/Typora_img/master/img/202204290943836.png" alt="image-20220403160826291" style="zoom:50%;" />

提取一句句子在另一句子中的相关语义作为注意力系数

<img src="https://raw.githubusercontent.com/akaBATMAN3/Typora_img/master/img/202204290943093.png" alt="image-20220403161108606" style="zoom:50%;" />

加强局部推理信息，捕获推理关系

#### Inference Composition

$$
v_a=BiLSTM(m_a) \\
v_b=BiLSTM(m_b)
$$

对局部推理信息再次利用BiLSTM编码，捕获其中的交互信息

#### Presion

<img src="https://raw.githubusercontent.com/akaBATMAN3/Typora_img/master/img/202204290943042.png" alt="image-20220403162121018" style="zoom:50%;" />

同时使用平均池化和最大池化，融合后送入分类器预测标签



### 实验结果

<img src="https://raw.githubusercontent.com/akaBATMAN3/Typora_img/master/img/202204290944114.png" alt="100epochs" style="zoom: 80%;" />

| Model    | Train acc | Test acc |
| -------- | --------- | -------- |
| 50D ESIM | 89.7      | 84.2     |

<img src="https://raw.githubusercontent.com/akaBATMAN3/Typora_img/master/img/202204290944453.png" alt="image-20220403134455547" style="zoom: 50%;" />	



## 任务四：基于LSTM+CRF的序列标注

### LSTM_CRF

```python
LSTM_CRF(
  (word_emb_layer): Embedding(26870, 50)
  (dropout): Dropout(p=0.5, inplace=False)
  (lstm): LSTM(50, 50, batch_first=True, bidirectional=True)
  (linear): Linear(in_features=100, out_features=12, bias=True)
  (crf): CRF()
)
```

#### CRF

利用转移矩阵$A$和BiLSTM提取的特征矩阵$P$计算损失
$$
loss=-log\ p(y|X)\\
p(y|X)=\frac{exp(s(X,y))}{\sum_{\tilde{y}\in Y_X}exp(s(X,\tilde{y}))}\\
Y_X\ represents\ all\ possible\ tag\ sequences\\
s(X,y)=\sum_{i=0}^nA_{y_i,y_{i+1}}+\sum_{i=1}^nP_{i,y_i}\\
A_{i,j}\ means\ the\ transfer\ score\ of\ tag\ i\ to\ tag\ j,\ P_{i,j}\ means\ the\ score\ of\ j^{th}\ tag\ of\ i^{th}\ word
$$

预测目标序列
$$
y^*=\underset{\tilde{y}\in Y_X}{argmax}\ s(X,\tilde{y})
$$
pytorch实现crf：https://towardsdatascience.com/implementing-a-linear-chain-conditional-random-field-crf-in-pytorch-16b0b9c4b4ea



### 实验结果

#### 训练集

<img src="https://raw.githubusercontent.com/akaBATMAN3/Typora_img/master/img/202204290945021.png" alt="100epochs-16511386562251" style="zoom:80%;" />

#### 测试集

| Model    | Acc  | F1   |
| -------- | ---- | ---- |
| LSTM_CRF | 96.1 | 90.6 |



## 任务五：基于神经网络的语言模型

### GRU

<img src="https://raw.githubusercontent.com/akaBATMAN3/Typora_img/master/img/202204290946489.png" alt="20epochs_gru" style="zoom: 80%;" />

```
春夏秋冬：
春，捷歌之櫜弭船二，二缚楚王直陪迎阵，秣骋骐驎。
夏谢意，金夫劳生樊圻士实，各保性固难。
秋心辱心，独去恨不见飞去然感，城压伍胥闲。
冬王至子预要，要非陈厌叫呼虞加，惟川鬼鬼雄。

我爱语言：
我。
爱骅口将灰，古不持终谟又谟楚臣二十超然五素质。
语数卢西西西江头西，江楚气八蟆又神襄襄泪，血污辞黄金口然超，
言女不热，只待官军亦驎。

古诗真美：
古隐，潇羞忍耻。
诗骅骝古死，丁库图邂塞枭古。
真痕耻是耻。
美人设网当金要相曹牵，丁人世人亭。
```

### LSTM

<img src="https://raw.githubusercontent.com/akaBATMAN3/Typora_img/master/img/202204290946920.png" alt="20epochs_lstm" style="zoom:80%;" />

```
春夏秋冬：
春妖炀，山河洛未祥。
夏君。
秋岸数具怆霭，山河转使莫相催。
冬上峡水，山公逐吐伸。

我爱语言：
我玩兵还人，山作岱宗钓登讲论陪喷墨，山河转使莫相催。
爱流不可求。
语槿遗兰醉春声，山楚未追春。
言涨未用，山公谷名愚挂席。

古诗真美：
古槿先吟道，山阴转使人。
诗倾森魁。
真槿绽红柱，山公助酣歌。
美事百献，山公鬼薄汉舞楼人人便，山公逐庭歌。
```
