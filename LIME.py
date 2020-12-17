# coding=utf-8

import numpy as np
import sklearn.metrics
from lime.lime_text import LimeTextExplainer
from sklearn.metrics import accuracy_score

from utils.utils import load_pkl

class_names = ['pos', 'neg']

train_x, test_x= load_pkl('./vector/train_seq_docs.pkl')['docs'], load_pkl('./vector/test_seq_docs.pkl')['docs']
train_y, test_y = np.array([0 if i < len(train_x)//2 else 1 for i in range(len(train_x))]),  np.array([0 if i < len(
    test_x)//2 else 1 for i in range(len(test_x))])

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(train_x)
test_vectors = vectorizer.transform(test_x)

rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train_vectors, train_y)


pred = rf.predict(test_vectors)
b = pred.copy()
b[b >= 0.5] = 1
b[b < 0.5] = 0
acc = accuracy_score(test_y, b)
a = sklearn.metrics.f1_score(test_y, pred, average='binary')
print('Acc, F1', acc, a)


from sklearn.pipeline import make_pipeline
c = make_pipeline(vectorizer, rf)

# print(c.predict_proba([test_x[80]]))


explainer = LimeTextExplainer(class_names=class_names)


idx = 80
exp = explainer.explain_instance(test_x[idx], c.predict_proba, num_features=10)
print('Document id: %d' % idx)
print('Probability(neg) =', c.predict_proba([test_x[idx]])[0,1])
print('True class: %s' % class_names[test_y[idx]])


print(exp.as_list())


print('Original prediction:', rf.predict_proba(test_vectors[idx])[0,1])
tmp = test_vectors[idx].copy()
tmp[0,vectorizer.vocabulary_['loved']] = 0
tmp[0,vectorizer.vocabulary_['cant']] = 0
print('Prediction removing some features:', rf.predict_proba(tmp)[0,1])
print('Difference:', rf.predict_proba(tmp)[0,1] - rf.predict_proba(test_vectors[idx])[0,1])



fig = exp.as_pyplot_figure()
fig.savefig('figure.jpg')

exp.save_to_file('oi.html')