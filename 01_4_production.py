# encoding: utf-8
# 朴素贝叶斯分类 iris, 使用持久化的模型进行预测

from sklearn.externals import joblib
# from sklearn.naive_bayes import GaussianNB


# 1、加载持久化模型
model = joblib.load('model_iris_GNB.pkl')
print(model)
''' GaussianNB(priors=None) '''

# 2、使用加载的模型进行预测
pred = model.predict([[ 5.1,  3.5,  1.4,  1.2]])
print(pred)
# [1]

