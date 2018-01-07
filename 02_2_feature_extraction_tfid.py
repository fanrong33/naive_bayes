# encoding: utf-8
# 文本数据的特征提取 TfidfVectorizer
 
# TfidfVectorizer方法
# 在CountVectorizer方法中，我们仅仅是统计了词汇的出现次数，比如单词the会出现比较多的次数，但实际上，这个单词并不是很有实际意义。因此使用TF-IDF方法来进行向量化操作。
# TF-IDF原本是一种统计方法，用以评估字词对于一个文件集或一个语料库中的其中一份文件的重要程度。
# 这个方法认为，字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降，
# 其实也就相当于在CountVectorizer的基础上结合整个语料库考虑单词的权重，并不是说这个单词出现次数越多它就越重要。


from sklearn.feature_extraction.text import TfidfVectorizer

text = [
    "The quick brown fox jumped over the lazy dog.",
    "The dog.",
    "The fox"
]

# create the transform
# stop_words = ['quick']
vectorizer = TfidfVectorizer(stop_words=[])
print(vectorizer)
'''
TfidfVectorizer(analyzer='word', binary=False, decode_error='strict',
        dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
        lowercase=True, max_df=1.0, max_features=None, min_df=1,
        ngram_range=(1, 1), norm='l2', preprocessor=None, smooth_idf=True,
        stop_words=None, strip_accents=None, sublinear_tf=False,
        token_pattern='(?u)\\b\\w\\w+\\b', tokenizer=None, use_idf=True,
        vocabulary=None)
'''
print(vectorizer.get_stop_words())


# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(vectorizer.vocabulary_)
'''
{'the': 7, 'quick': 6, 'brown': 0, 'fox': 2, 'jumped': 3, 'over': 5, 'lazy': 4, 'dog': 1}
'''
print(vectorizer.get_feature_names())
'''
['brown', 'dog', 'fox', 'jumped', 'lazy', 'over', 'quick', 'the']
'''
print(vectorizer.idf_)
'''
[ 1.69314718  1.28768207  1.28768207  1.69314718  1.69314718  1.69314718
  1.69314718  1.        ]
'''

# encode document
vector = vectorizer.transform([text[0]])
# summarize encoded vector
print(vector.shape)
''' (1, 8) '''
print(vector.toarray())
'''
[[ 0.36388646  0.27674503  0.27674503  0.36388646  0.36388646  0.36388646
   0.36388646  0.42983441]]
'''
vector = vectorizer.transform([text[1]])
print(vector.toarray())
'''
[[ 0.          0.78980693  0.          0.          0.          0.          0.
   0.61335554]]
'''

'''
这里是使用了3个小文本，总共有8个词汇。可以看到单词the在每个文本中都出现过，这就导致他的idf值是最低的，显得单词the并没那么重要。
通过输出前两个文本的向量对比，可以发现这种表示方法依然会输出稀疏向量。

'''

