import datetime
import numpy as np
import scipy as sp
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.utils.multiclass import type_of_target
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from  pprint import pprint
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from IPython.display import Image
from sklearn.externals.six import StringIO
import pydotplus
#初始化#
starttime=datetime.datetime.now()
data=[]
labels=[]
result=[]

#读入数据
with open('1.txt')as ifile:
    for line in ifile:
        tokens=line.strip().split(' ')
        data.append([float(tk)for tk in tokens[:-1]])
        labels.append(tokens[-1])
x=np.array(data)
y=np.array(labels)
#参数选择

parameters={}
clf=tree.DecisionTreeClassifier()
clf=GridSearchCV(clf,parameters)
clf.fit(x,y)
print(clf.best_estimator_)
result_sum=[]
result_train=[]
result_test=[]
r_test=[]
r_train=[]
#模型训练
clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-08, min_samples_leaf=1,
            min_samples_split=30, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')
for i in range(5):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    clf.fit(x_train,y_train)
    # result.append(np.mean(y_test==clf.predict(x_test)))#测试集中实际值和预测值对比的总精度#
    result_train.append(clf.score(x_train, y_train))  # 训练集平均精度#
    for j in range(5):
        r_train.append(result_train)
        avg1 = np.mean(r_train)
    result_test.append(clf.score(x_test, y_test))  # 测试集平均精度#
    for j in range(5):
        r_test.append(result_test)
        avg2 = np.mean(r_test)

#输出结果
print("训练集平均精度",avg1)
print("测试集平均精度",avg2)
print("稳健性",abs(avg1-avg2))
y_pred=clf.predict(x)
m=confusion_matrix(y,clf.predict(x))
print("混淆矩阵",m)
print("总分类精度",((m[1,1]+m[0,0]))/(m.sum()))
print("输出各个参数的权重")
print(metrics.classification_report(y,y_pred))