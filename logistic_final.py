#导入相关库函数
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from  pprint import pprint
from sklearn.metrics import confusion_matrix
#初始化
starttime=datetime.datetime.now()
data=[]
labels=[]
result=[]
result_train=[]
result_test=[]
r_train=[]
r_test=[]
#数据读入#
with open('1.txt')as ifile:
    for line in ifile:
        tokens=line.strip().split(' ')
        data.append([float(tk)for tk in tokens[:-1]])
        labels.append(tokens[-1])
x=np.array(data)
y=np.array(labels)

# #PCA降维
# pca=PCA(n_components=18)
# reduced_X=pca.fit_transform(x)

#测试集与训练集区分
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
#标准化
sc = StandardScaler()
sc.fit(X_train)
X_std=sc.transform(x)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#五折交叉验证，模型训练
for i in range(5):
    model = LogisticRegression(C=1000.0, random_state=0)
    model.fit(X_train_std,y_train)
    test_prob = model.predict(X_test_std)
    result.append(np.mean(y_test==test_prob))#实际值和预测值对比的总精度#
    result_train.append(model.score(X_train_std, y_train))  # 训练集平均精度#
    for j in range(5):
        r_train.append(result_train)
        avg = np.mean(r_train)
    result_test.append(model.score(X_test_std, y_test))  # 测试集平均精度#
    for j in range(5):
        r_test.append(result_test)
        avg2 = np.mean(r_test)
print("logistic classifier accuacy:", result)

#输出结果
print("训练集精度",result_train)
print("测试集精度",result_test)
print("训练集平均精度",avg)
print("测试集精度",result_test)
print("测试集平均精度",avg2)
print("稳健性",abs(avg-avg2))
print("输出各个参数的权重")
print("混淆矩阵",confusion_matrix(y_test,model.predict(X_test_std)))
print(model.coef_)
h=model.coef_[0]
y_pred=model.predict(X_std)
print(metrics.classification_report(y,y_pred))

#绘图
k=[i for i in range(1,25)]
plt.bar(k,h)
plt.xticks(k, (str(i) for i in range(1, 25)))
plt.title('index weight')
plt.xlabel('sequence')
plt.ylabel('weight')
plt.show()

#运行时间
endtime=datetime.datetime.now()
time=(endtime-starttime).seconds
print("the running time of this progarmme is",time,'s')