# Natural Language Inference based on Attention Mechanism

## 1	ESIM结构

<img src="https://raw.githubusercontent.com/akaBATMAN3/nlp-beginner/master/imgs/202205111737131.png" alt="image-20220403155034356" style="zoom: 67%;" />

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

### 1.1	Input Encoding

$$
\overline{a}_i=BiLSTM(a,i),\forall i\in[1,...,l_a]\\
\overline{b}_i=BiLSTM(b,i),\forall i\in[1,...,l_b]
$$

将句子用词嵌入表示，之后两句句子的每个单词的词嵌入经过BiLSTM提取特征，用提取的隐藏状态作为单词上下文特征。

### 1.2	Local Inference Modeling

$$
e_{ij}=\overline{a}_i^T\overline{b}_j
$$

软对齐层计算两句句子之间的相似度作为注意力权重。
$$
\tilde{a}_i=\sum_{j=1}^{l_b}\frac{exp(e_{ij})}{\sum_{k=1}^{l_b}exp(e_{ik})}\overline{b}_j,\forall i\in[1,...,l_a]\\
\tilde{b}_j=\sum_{i=1}^{l_a}\frac{exp(e_{ij})}{\sum_{k=1}^{l_a}exp(e_{kj})}\overline{a}_i,\forall i\in[1,...,l_b]\\
$$
提取一句句子在另一句子中的相关语义作为注意力系数。
$$
m_a=[\overline{a};\tilde{a};\overline{a}-\tilde{a};\overline{a}\odot\tilde{a}]\\
m_b=[\overline{b};\tilde{b};\overline{b}-\tilde{b};\overline{b}\odot\tilde{b}]
$$
加强局部推理信息，捕获推理关系。

### 1.3	Inference Composition

$$
v_a=BiLSTM(m_a) \\
v_b=BiLSTM(m_b)
$$

对局部推理信息再次利用BiLSTM编码，捕获其中的交互信息。

### 1.4	Prediction

$$
v_{a,ave}=\sum_{i=1}^{l_a}\frac{v_{a,i}}{l_a},v_{a,max}=\mathop{\max}_{i=1}^{l_a}v_{a,i}\\
v_{b,ave}=\sum_{j=1}^{l_b}\frac{v_{b,i}}{l_b},v_{b,max}=\mathop{\max}_{j=1}^{l_b}v_{b,i}\\
v=[v_{a,ave};v_{a,max};v_{b,ave};v_{b,max}]
$$

同时使用平均池化和最大池化，融合后送入分类器预测标签。

## 2	实验结果

​	训练集上表现。

<img src="https://raw.githubusercontent.com/akaBATMAN3/nlp-beginner/master/imgs/202205111736269.png" alt="100epochs" style="zoom: 67%;" />

​	与ESIM原文中的各模型比对。

| Model    | Train acc | Test acc |
| -------- | --------- | -------- |
| 50D ESIM | 89.7      | 84.2     |

<img src="https://raw.githubusercontent.com/akaBATMAN3/nlp-beginner/master/imgs/202205111736400.png" alt="image-20220403134455547" style="zoom: 50%;" />	

## 参考

[1] [Enhanced LSTM for Natural Language Inference](https://arxiv.org/pdf/1509.06664v1.pdf)

[2] [Reasoning about Entailment with Neural Attention](https://arxiv.org/pdf/1509.06664v1.pdf)