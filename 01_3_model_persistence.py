# encoding: utf-8
# 朴素贝叶斯分类 iris, 并进行模型持久化

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn import datasets
from sklearn.externals import joblib


# 1、加载数据集
#我们假定sepal length, sepal width, petal length, petal width 4个量独立且服从高斯分布，用贝叶斯分类器建模
iris = datasets.load_iris()
print(iris.data)
'''
array([[ 5.1,  3.5,  1.4,  0.2],
       [ 4.9,  3. ,  1.4,  0.2],
       [ 4.7,  3.2,  1.3,  0.2],
       [ 4.6,  3.1,  1.5,  0.2],
       [ 5. ,  3.6,  1.4,  0.2]])
'''
print(iris.target[:5])
''' [0 0 0 0 0] '''


# 2、划分数据集为训练集和测试集
seed = 2
test_size = 0.3 # 7:3
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=test_size, \
    random_state=seed)


# 3、训练模型
model = GaussianNB()
model.fit(X_train, y_train)
print(model)
''' GaussianNB(priors=None) '''


# 4、预测测试数据集
y_pred = model.predict(X_test)


# 5、评价预测的正确率
accuracy = accuracy_score(y_test, y_pred)
print("Naive Bayes Accuracy: %.2f%%" % (accuracy*100.0))
''' Naive Bayes Accuracy: 97.78% '''


# 6、将训练好的模型持久化
import os
abspath = os.path.abspath(__file__)
path = os.path.split(abspath)[0]
filenames = joblib.dump(model, os.path.join(path, 'model_iris_GNB.pkl'))
print(filenames)
