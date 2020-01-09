import matplotlib.pyplot as plt
import matplotlib.axes as ax

import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

from MultiSVM import *
from mcsvm import CSSVM
from mesh import *


N=30
s = np.random.normal(0, 0.5,(N,2))

X1=s+[0,0]
X2=s+[0,6]
X3=s+[6,6]
X4=s+[6,0]


y=np.concatenate([0*np.ones(N),1*np.ones(N),2*np.ones(N),3*np.ones(N)])
y=y.astype(int)
X=np.concatenate([X1,X2,X3,X4])
#weight_sample = np.concatenate([1*np.ones(2*N),10*np.ones(2*N)])
#theta=np.array([10,10,10,1],dtype = np.int)
class_weight={0:1,1:1,2:10,3:10}
MyCls=MultiSVM(kernel='linear',class_weight=class_weight)
MyCls.fit(X,y)

plt.clf()
plt.plot(X1[:,0],X1[:,1],'*r',X2[:,0],X2[:,1],'^b',X3[:,0],X3[:,1],'+y',X4[:,0],X4[:,1],'og')


x = np.arange(-2, 10)
uni=MyCls.class_weight
w=MyCls.w2#.value
y1 = -(w[0, 0] * x + w[0,2] -uni[0] ) / w[0, 1]
plt.plot(x, y1.T, color='red')

y2 = -(w[1, 0] * x + w[1,2] -uni[1]) / w[1, 1]
plt.plot(x, y2.T, color='blue')

y3 = -(w[2, 0] * x + w[2,2]-uni[2]) / w[2, 1]
plt.plot(x, y3.T, color='yellow')

y4 = -(w[3, 0] * x + w[3,2]-uni[3]) / w[3, 1]
plt.plot(x, y4.T, color='green')

xx, yy = make_meshgrid(np.array([-2,10]), np.array([-2,10]))
plot_contours(plt, MyCls, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)


clf1 = svm.SVC(kernel='linear',decision_function_shape='ovo',class_weight=class_weight)
clf2 = svm.SVC(kernel='linear',decision_function_shape='ovr',class_weight=class_weight)
clf3 = svm.LinearSVC(multi_class='crammer_singer',class_weight=class_weight)
#clf3 = CSSVM(kernel='poly',class_weight=class_weight)





model = (clf1,clf2,clf3,MyCls)

models = (clf.fit(X, y) for clf in model)
# title for the plots
titles = ('CSOVO',
          'CSOVA',
          'CSCS',
          'Apportioned SVM')

# Set-up 2x2 grid for plotting.
#plt.figure()
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

xx, yy = make_meshgrid(X[:, 0], X[:, 1])

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    # ax.set_xlabel('x label')
    # ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()



