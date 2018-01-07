# encoding: utf-8
# 简单文本分类示例
# 
# 机器学习中，我们总是要先将源数据处理成符合模型算法输入的形式，比如将文字、声音、图像转化成矩阵。
# 对文本数据首先要进行分词（tokenization），移除停止词（stop words），然后将词语转化成矩阵形式，
# 然后再输入机器学习模型中，这个过程称为特征提取（feature extraction）或者向量化（vectorization）。


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# 1、加载数据集
corpus = [
    "The football play",    # sample 1
    "The tencent wangzhe.", # sample 2
    "The good world girl"   # sample 3
]
target = ['sport','game','news']


# 2、预处理
# create the transform
vectorizer = TfidfVectorizer()
# tokenize and build vocab
vectorizer.fit(corpus)

# encode document
X = vectorizer.transform(corpus).toarray()
print(X)
'''
[[ 0.65249088  0.          0.          0.65249088  0.          0.38537163
   0.          0.        ]
 [ 0.          0.          0.          0.          0.65249088  0.38537163
   0.65249088  0.        ]
 [ 0.          0.54645401  0.54645401  0.          0.          0.32274454
   0.          0.54645401]]
'''

# 编码字符串分类值为整形
# TODO 
label_encoder = LabelEncoder()
label_encoder_y = label_encoder.fit_transform(target)
''' 相当于
label_encoder.fit(target)
label_encoder_y = label_encoder.transform(target)
'''
print(label_encoder_y)
''' [2 0 1] '''




# 3、训练模型
model = GaussianNB()
model.fit(X, label_encoder_y) #特征数据直接灌进来


# 4、预测测试数据集
y_pred = model.predict(X)


# 5、评价预测的正确率
accuracy = accuracy_score(label_encoder_y, y_pred)
print("Naive Bayes Accuracy: %.2f%%" % (accuracy*100.0))


corpus = ['love play ball']
vector = vectorizer.transform(corpus)
pred = model.predict(vector.toarray())
print(pred)
''' [2] '''
print(label_encoder.inverse_transform(pred))
''' ['sport'] '''

