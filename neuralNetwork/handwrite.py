import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import LabelBinarizer
from neuralNetwork.nn import NeturalNetwork
from sklearn.cross_validation import train_test_split

digits =load_digits()
#1797张 8*8的手写数字图片
X = digits.data
Y = digits.target
#标准化
X -=X.min()
X /=X.max()

nn = NeturalNetwork([64,100,10],'logistic')

#分离测试集跟训练集
X_train,X_test,y_train,y_test = train_test_split(X,Y)
labels_train = LabelBinarizer().fit_transform(y_train)
labels_test  = LabelBinarizer().fit_transform(y_test)

print("start fitting")

nn.fit(X_train,labels_train,epochs=3000)

print("end training")
predictions=[]

for i in range(X_test.shape[0]):
    o = nn.predict(X_test[i])
    predictions.append(np.argmax(o))

#10*10矩阵（分类是10） ，对角线表示预测的对，行是预测值，列是真实值
print(confusion_matrix(y_test,predictions))
'''
v[[43  0  0  0  0  0  0  0  0  0]
 [ 0 37  0  0  0  0  1  0  0  8]
 [ 0  1 38  3  0  0  0  0  0  0]
 [ 0  0  1 47  0  1  0  1  0  0]
 [ 0  0  0  0 47  0  0  0  0  1]
 [ 0  0  0  0  0 48  1  0  0  0]
 [ 0  2  0  0  0  0 38  0  0  0]
 [ 0  0  0  0  1  0  0 37  0  1]
 [ 1  7  0  1  0  4  1  0 26  6]
 [ 0  0  0  4  0  0  0  0  0 43]]
'''
print(classification_report(y_test,predictions))
'''统计
             precision    recall  f1-score   support

          0       0.98      1.00      0.99        43
          1       0.79      0.80      0.80        46
          2       0.97      0.90      0.94        42
          3       0.85      0.94      0.90        50
          4       0.98      0.98      0.98        48
          5       0.91      0.98      0.94        49
          6       0.93      0.95      0.94        40
          7       0.97      0.95      0.96        39
          8       1.00      0.57      0.72        46
          9       0.73      0.91      0.81        47

avg / total       0.91      0.90      0.90       450

'''