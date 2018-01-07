# encoding: utf-8

import sys
from sklearn.externals import joblib
# from sklearn.naive_bayes import GaussianNB

text = []
if len(sys.argv) == 2:
    # 03_2_text_classification_production.py 'news content'
    text = sys.argv[1]
    text = [text]

if not text:
    text = [
'''
　　据5月8日海外消息，三星电子宣布将投资3.126亿美元来提升系统芯片的研发和制造能力。
　　三星电子在半导体业务中多以存储芯片为主，而系统芯片研发、加工在三星电子半导体业务中还只占很小的份额。据统计，今年一季度，来自系统芯片的收入仅占三星半导体业务收入的11％。现在三星加大对系统芯片制造业务的投资，主要是希望能满足日益增长的市场需要，并同时提高价格竞争力。"
　　在系统芯片领域，三星将面临着台积电、台联电、新加坡特许半导体等强劲对手的竞争。
'''
]

# 1、加载持久化模型
model = joblib.load('model_text_cla.pkl')
# print(model)
''' GaussianNB(priors=None) '''


vectorizer = joblib.load('vectorizer.pkl')
# print(vectorizer)
'''
TfidfVectorizer(analyzer='word', binary=False, decode_error='strict',
        dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
        lowercase=True, max_df=1.0, max_features=None, min_df=1,
        ngram_range=(1, 1), norm='l2', preprocessor=None, smooth_idf=True,
        stop_words=['\u3000'], strip_accents=None, sublinear_tf=False,
        token_pattern='(?u)\\b\\w\\w+\\b', tokenizer=None, use_idf=True,
        vocabulary=None)
'''


# 2、使用加载的模型进行预测
pred = model.predict(vectorizer.transform(text).toarray())
print(pred)
category_map = {
    'C000007': '汽车',
    'C000008': '财经',
    'C000010': 'IT',
    'C000013': '健康',
    'C000014': '体育',
    'C000016': '旅游',
    'C000020': '教育',
    'C000022': '招聘',
    'C000023': '文化',
    'C000024': '军事'
}
print(category_map[pred[0]])

