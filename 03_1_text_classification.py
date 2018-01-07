# encoding: utf-8
# 使用朴素贝叶斯对新闻分类，并进行模型持久化

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib


# 加载数据集
'''
[搜狐新闻数据](http://www.sogou.com/labs/resource/list_news.php)
数据介绍：来自搜狐近20个栏目的分类新闻数据。
发布时间：2012年8月16号
'''
import os
import codecs

class_code = [
    'C000007',
    'C000008',
    'C000010',
    'C000013',
    'C000014',
    'C000016',
    'C000020',
    'C000022',
    'C000023',
    'C000024']
corpus = []
seg_text = []
Y = []

for category in class_code:
    i=0
    filenames=os.listdir('SogouC/ClassFile/%s/'% category)
    for filename in filenames:
        i+=1
        if i<=200:
            try:
                with codecs.open('SogouC/ClassFile/%s/%s' % (category, filename), 'r', 'gb2312') as fp: # 注意：这里不是file.open()，没有file这个对象？
                    content = fp.read()
                    corpus.append(content)
                    Y.append(category)
            except:
                pass

# 新闻文本分词
print('对新闻分词')
import jieba
for i in corpus:
    seg_list = jieba.cut(i)
    seg_text.append(' '.join(seg_list))

# 特征提取
# \u3000 全角空格
# 去掉停用词



print('特征提取')
stop_words= ['\u3000']
# TODO 完善中文停用词
vectorizer = TfidfVectorizer(stop_words=stop_words)
vectorizer.fit(seg_text)
X = vectorizer.transform(seg_text).toarray()


# 训练模型
print('训练模型')
model = GaussianNB()
model.fit(X, Y)
print(X.shape)

joblib.dump(model, 'model_text_cla.pkl')


joblib.dump(vectorizer, 'vectorizer.pkl')


# 预测测试数据集
print('预测测试数据集')
y_pred = model.predict(X)
print(y_pred)

# 5、评价预测的正确率
accuracy = accuracy_score(Y, y_pred)
print("Naive Bayes Accuracy: %.2f%%" % (accuracy*100.0))


# 使用训练好的模型进行预测
text = [
'''
　　据5月8日海外消息，三星电子宣布将投资3.126亿美元来提升系统芯片的研发和制造能力。
　　三星电子在半导体业务中多以存储芯片为主，而系统芯片研发、加工在三星电子半导体业务中还只占很小的份额。据统计，今年一季度，来自系统芯片的收入仅占三星半导体业务收入的11％。现在三星加大对系统芯片制造业务的投资，主要是希望能满足日益增长的市场需要，并同时提高价格竞争力。"
　　在系统芯片领域，三星将面临着台积电、台联电、新加坡特许半导体等强劲对手的竞争。
'''
]
vector = vectorizer.transform(text)
pred = model.predict(vector.toarray())
print(pred)




