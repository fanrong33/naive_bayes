# encoding: utf-8
# 朴素贝叶斯分类 iris

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn import datasets

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
print(iris.data.shape)
''' (150, 4) # 150行4列 '''
print(iris.target[:5])
''' [0 0 0 0 0] '''


# 2、训练模型
model = GaussianNB()
model.fit(iris.data, iris.target)
print(model)
''' GaussianNB(priors=None) '''

# 3、预测测试数据
y_pred = model.predict(iris.data)


# 4、评价预测的正确率
accuracy = accuracy_score(iris.target, y_pred)
print("Naive Bayes Accuracy: %.2f%%" % (accuracy*100.0))
''' Naive Bayes Accuracy: 96.00% '''


# 5、log损失
# 在这个多分类问题中，Kaggle的评定标准并不是precision，而是multi-class log_loss，这个值越小，表示最后的效果越好
predicted = model.predict_proba(iris.data)
print("Naive Bayes log loss: %f " % (log_loss(iris.target, predicted)))
''' Naive Bayes log loss: 0.111249 '''


# 6、使用模型进行预测
pred = model.predict([[5., 3.2, 1.3, 0.3]])
print(pred)

