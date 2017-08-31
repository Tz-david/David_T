import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import requests
import urllib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.datasets import make_classification
from sklearn import metrics
import seaborn
from pprint import pprint
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
data=[]
labels=[]

#数据预处理
#数据读入#
with open('1.txt')as ifile:
    for line in ifile:
        tokens=line.strip().split(' ')
        data.append([float(tk)for tk in tokens[:-1]])
        labels.append(tokens[-1])
x=np.array(data)
y=np.array(labels)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#模型
estimator=RandomForestClassifier(bootstrap=True, criterion='gini', max_depth=8.75, max_features='auto',
                                 max_leaf_nodes=None,  min_samples_leaf=1,min_samples_split=2, n_estimators=1000,
                                 n_jobs=1, oob_score=False,
                                 random_state=None, verbose=0)
estimator.fit(x_train,y_train)
print(estimator.feature_importances_)

importances=estimator.feature_importances_
std=np.std([tree.feature_importances_ for tree in estimator.estimators_],axis=0)
indices=np.argsort(importances)[::-1]
print("Top 10 Features:")
indices=indices[:10]
plt.figure()
plt.title("Top 10 feature importances")
plt.bar(range(10),importances[indices],color='r',yerr=std[indices],align="center")
plt.xticks(range(10),indices)
plt.xlim([-1,10])


x_train_r=estimator.transform(x_train,threshold='mean')
x_test_r=estimator.transform(x_test,threshold='mean')
x_r=estimator.transform(x,threshold='mean')
estimator.fit(x_train_r,y_train)
y_pred=estimator.predict(x_test_r)
print(metrics.classification_report(y_test,y_pred))

k1=estimator.score(x_train_r,y_train)
k2=estimator.score(x_test_r,y_test)

print(k1,k2)
m=confusion_matrix(y,estimator.predict(x_r))
print("混淆矩阵",m)
print("总分类精度",((m[1,1]+m[0,0]))/(m.sum()))

plt.show()

