# Sequence Labeling based on LSTM+CRF

## 1	LSTM_CRF

```python
LSTM_CRF(
  (word_emb_layer): Embedding(26870, 50)
  (dropout): Dropout(p=0.5, inplace=False)
  (lstm): LSTM(50, 50, batch_first=True, bidirectional=True)
  (linear): Linear(in_features=100, out_features=12, bias=True)
  (crf): CRF()
)
```

### 1.1	CRF

​	利用转移矩阵$A$和BiLSTM提取的特征矩阵$P$计算损失
$$
loss=-log\ p(y|X)\\
p(y|X)=\frac{exp(s(X,y))}{\sum_{\tilde{y}\in Y_X}exp(s(X,\tilde{y}))}\\
s(X,y)=\sum_{i=0}^nA_{y_i,y_{i+1}}+\sum_{i=1}^nP_{i,y_i}\\
$$

其中$Y_X$表示所有可能的标签序列，$A_{i,j}$表示从标签$i$转移到标签$j$的分数，$P_{i,j}$表示第$i$个单词的第$j$个标签的分数。

​	预测目标序列
$$
y^*=\underset{\tilde{y}\in Y_X}{argmax}\ s(X,\tilde{y})
$$
pytorch实现crf：https://towardsdatascience.com/implementing-a-linear-chain-conditional-random-field-crf-in-pytorch-16b0b9c4b4ea.

## 2	实验结果

​	训练集上的表现

<img src="https://raw.githubusercontent.com/akaBATMAN3/nlp-beginner/master/imgs/202205111742086.png" alt="100epochs-16511386562251" style="zoom: 67%;" />

​	测试集上的表现。

| Model    | Acc  | F1   |
| -------- | ---- | ---- |
| LSTM_CRF | 96.1 | 90.6 |

## 参考

[1] [Neural Architectures for Named Entity Recognition](https://arxiv.org/pdf/1603.01360.pdf)

[2] [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](https://arxiv.org/pdf/1603.01354.pdf)

