# Sentence Similarity Calculation

## Task description

​	语义相似度计算是许多文本信息处理应用（比如文本分类或文本聚类）的基础。常用的相似度计算方法有余弦相似度、欧式距离和编辑距离等。本题目要求各位同学实现计算两个句子语义相似度的简单方法，并能评价自己所实现的方法的效果，要求如下：

1）文件input.txt中包含了750个句子对（每行包含一个句子对，句子对中间的分割符为tab键）。请实现利用余弦相似度计算这750个句子对的语义相似度的程序，并给出每个句子对的相似度计算结果。相似度计算的结果文件（output.txt）应包括750行（顺序与输入文件相同），每行的内容包括句子对及其对应的相似度值。格式如下：相似度值+“tab键”+句子1+“tab键”+句子2

2）在第1）步中，我们计算了750个句子对的相似度。如何评价我们的计算结果呢？可以用Pearson correlation来衡量两个打分序列的相关性。因此，如果我们有这750个句子对的人工打分结果，就可以使用Pearson correlation来衡量我们方法的结果和人工打分结果之间的相关度。如果这个相关度很高，可以在一定程度上说明计算机所得的相似度结果和人工标注的结果比较一致。golden.txt文件中保存的是input.txt中的750个句子对的人工打分结果。这个文件也包含750行，每行对应一个句对的相似度值，句子对顺序与input.txt中的相同。请计算你实现方法的结果（output.txt）与人工打分结果之间的Pearson correlation。

## Usage

1. Download and preprocess pretrained word embedding (see [pretrained_emb/README.md](./pretrained_emb/README.md))

1. Download pretrained InferSent model from [encoder/README.md](./encoder/README.md)

1. Run the following code to get needed nltk dependencies

   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   nltk.download('punkt')
   ```

1. Run main.py

1. Check results in out directory

1. More details in [report](./report.md)