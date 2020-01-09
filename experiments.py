from sklearn import svm
from sklearn.datasets import *
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import *
from pandas import read_csv

from MultiSVM import *

# from sklearn import datasets
dataset = load_iris()
#dataset = load_wine()
# dataset = load_digits()
# #
X = dataset.data
y = dataset.target
#class_weight={0:2,1:1,2:1}

#X, y=load_svmlight_files(["assets/vehicle.scale"])

#y=y.astype(int)-1

#for glass
#y[y==6]=3

#for diabetes # / german / heart
#y=((y+1)/2).astype(int)

#for breast
#y=(y/2).astype(int)-1

#class_weight={0:2,1:1,2:1,3:1,4:1,5:1}
class_weight={0:2,1:1,2:1}#,3:1,4:1,5:1,6:1,7:1,8:1,9:1}

kernel='rbf'

param_grid = {'C': [0.1,0.5,1,5,10,50,100],
             'degree': [1,2,3,4,5,6,7,8,9] }





clf1=MultiSVM(kernel=kernel,class_weight=class_weight)
clf2 = svm.SVC(kernel=kernel,decision_function_shape='ovr',class_weight=class_weight)
clf3 = svm.LinearSVC(multi_class='crammer_singer',class_weight=class_weight)
#clf3 = CSSVM(multi_class='crammer_singer',class_weight=class_weight)
clf4 = svm.SVC(kernel=kernel,decision_function_shape='ovo',class_weight=class_weight)

titles=["multiSVM","ovr","cramer","ovo"]
models = (clf1,clf2,clf3,clf4)
for clf1 , title in zip(models,titles):
   # clf = GridSearchCV(clf1, param_grid, cv=5)
    y_pred = cross_val_predict(clf1, X, y, cv=2)
    conf_mat = confusion_matrix(y, y_pred)
    sum=0
    for i in range(len(conf_mat)):
        sum += (y[y ==i].shape[0]-conf_mat[i][i])*class_weight[i]
    print(title, sum/X.shape[0])

    print(title,classification_report(y, y_pred))

#
# def validate(clf,cv):
#     score = 0
#     for i in range(cv):
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
#         #models = (clf.fit(X_train, y_train) for clf in model)
#
#         clf.fit(X_train,y_train)
#         # y_pred = clf1.predict(X_test)
#         # print(confusion_matrix(y_test,y_pred))
#         score +=clf1.score(X_test,y_test)
#
#     return score /cv


