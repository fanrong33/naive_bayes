# encoding: utf-8
# 文本数据的特征提取 CountVectorizer
# [机器学习中，使用Scikit-Learn简单处理文本数据](https://www.jianshu.com/p/e71bcfd02827)


from sklearn.feature_extraction.text import CountVectorizer

'''
机器学习中，我们总是要先将源数据处理成符合模型算法输入的形式，比如将文字、声音、图像转化成矩阵。
对于文本数据首先要进行分词（tokenization），移除停止词（stop words），然后将词语转化成矩阵形式，
然后再输入机器学习模型中，这个过程称为特征提取（feature extraction）或者向量化（vectorization）。
本文会教你使用Scikit-Learn机器学习库中的三种模型来实现这一转化过程，包括CountVectorizer, TfidfVectorizer, HashingVectorizer。

文本到数值域的特征抽取方式
word2vec
用互信息提取关键字
在文本检索系统中非常有效的一种特征：TF-IDF(term frequency-interdocument frequency)向量。

TF-IDF是一种统计方法，用以评估一字词(或者n-gram)对于一个文件集或一个语料库中的其中一份文件的重要程度。
字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。
'''

'''
CountVectorizer是通过fit_transform函数将文本中的词语转换为词频矩阵
get_feature_names() 可看到所有文本的关键字
vocabulary_         可看到所有文本的关键字和其位置
toarray()           可看到词频矩阵的结果
'''

# list of text documents 文本集，（注意：文本中分词以空格分隔）
corpus = [
    "The quick brown fox jumped over the lazy dog.",
]

# create the transform
vectorizer = CountVectorizer()

# tokenize and build vocab
# fit(.)函数从文档中学习出一个词汇表
vectorizer.fit(corpus)

# summarize
print(vectorizer.vocabulary_)
''' {'the': 7, 'quick': 6, 'brown': 0, 'fox': 2, 'jumped': 3, 'over': 5, 'lazy': 4, 'dog': 1} '''

print(vectorizer.get_feature_names())
''' ['brown', 'dog', 'fox', 'jumped', 'lazy', 'over', 'quick', 'the'] '''

# encode document
# transform(.)函数将指定文档转化为向量。
vector = vectorizer.transform(corpus)
print(vector)
'''
  (0, 0)    1
  (0, 1)    1
  (0, 2)    1
  (0, 3)    1
  (0, 4)    1
  (0, 5)    1
  (0, 6)    1
  (0, 7)    2
'''
# summarize encoded vector
print(vector.shape)
''' (1, 8) '''
print(type(vector))
''' <class 'scipy.sparse.csr.csr_matrix'> '''
print(vector.toarray())
''' [[1 1 1 1 1 1 1 2]] '''


corpus2 = ["the puppy"]
vector = vectorizer.transform(corpus2)
print(vector.toarray())
''' [[0 0 0 0 0 0 0 1]] '''

