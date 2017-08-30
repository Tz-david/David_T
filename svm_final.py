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

#模型训练
clf_linear=svm.SVC(kernel='linear')
result_train=[]
result_test=[]
r_test=[]
r_train=[]
for i in range(5):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    clf_linear.fit(x_train,y_train)
    result.append(np.mean(y_test==clf_linear.predict(x_test)))#实际值和预测值对比的总精度#
    result_train.append(clf_linear.score(x_train,y_train))#训练集平均精度#
    for j in range(5):
        r_train.append(result_train)
        avg=np.mean(r_train)
    result_test.append(clf_linear.score(x_test, y_test))#测试集平均精度#
    for j in range(5):
        r_test.append(result_test)
        avg2=np.mean(r_test)
print("svm classifier accuacy:",result)
print("训练集精度",result_train)
print("测试集精度",result_test)
print("训练集平均精度",avg)
print("测试集精度",result_test)
print("测试集平均精度",avg2)
print("稳健性",abs(avg-avg2))
y_pred=clf_linear.predict(x)
print("输出各个参数的权重")
print("混淆矩阵",confusion_matrix(y_test,clf_linear.predict(x_test)))
print(clf_linear.coef_)
h=clf_linear.coef_[0]

#输出结果
print(metrics.classification_report(y,y_pred))

#绘图
x=[i for i in range(1,25)]
plt.bar(x,h)
plt.xticks(x, (str(i) for i in range(1, 25)))
plt.title('index weight')
plt.xlabel('sequence')
plt.ylabel('weight')
plt.show()

#运行时间
endtime=datetime.datetime.now()
time=(endtime-starttime).seconds
print("the running time of this progarmme is",time,'s')