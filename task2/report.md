# Text Classification based on Deep learning

## 1	文本特征表示方法

​	word embedding是将每个单词映射到高维空间的向量，其中不同维度可能代表不同真实世界含义。由于每个维度都由连续值来表示，因此词向量间可能产生相关性（如在每个维度上靠近彼此或一组组合有相似对立关系）。word embedding需要由训练获得，可以随机初始化或采用预训练好的word embedding。

## 2	Dropout

​	在每次前向/反向传播中按照一定概率（1-p）随机drop一些神经元子集，测试阶段用全部神经元来预测

## 3	Text_RNN

```python
Text_RNN(
  (word_emb_layer): Embedding(16532, 50)
  (dropout): Dropout(p=0.2, inplace=False)
  (rnn): RNN(50, 50, batch_first=True)
  (linear): Linear(in_features=50, out_features=5, bias=True)
)
```

## 4	Text_CNN

使用n*word embedding size的卷积核扫描由word embedding组成的一句句子，其中n可以设为2，3，4来挖掘词组关系。以下为TextCNN结构。

<img src="https://raw.githubusercontent.com/akaBATMAN3/nlp-beginner/master/imgs/202205111707475.png" alt="image-20220319170235075" style="zoom: 67%;" />

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

## 5	实验结果

​	训练集上不同方法的表现。

<img src="https://raw.githubusercontent.com/akaBATMAN3/nlp-beginner/master/imgs/202205111708984.png" alt="50epochs_comparison" style="zoom:67%;" />

​	最终测试集上结果见下表，Long sent acc为长度大于25的句子的准确率。

| 模型       | Acc      | Long sent acc |
| ---------- | -------- | ------------- |
| random_rnn | 57.8     | 46.3          |
| random_cnn | 56.6     | 47.1          |
| glove_rnn  | **63.8** | 49.1          |
| glove_cnn  | 59.6     | **51.5**      |

## 参考

[1] [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

