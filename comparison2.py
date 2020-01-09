import matplotlib.pyplot as plt
import numpy as np
from cvxopt import *
from cvxpy import *
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

from MultiSVM import *
from mesh import *

N=20
s = np.random.normal(0, 0.5,(N,2))

X1=s+[0,0]
X2=s+[0,6]
X3=s+[6,6]
X4=s+[6,0]


fig, sub = plt.subplots(2,2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
cls = MultiSVM(kernel='linear')


X=np.concatenate([X1,X2,X3,X4])
y = np.concatenate([0 * np.ones(N), 1 * np.ones(N), 2 * np.ones(N), 3 * np.ones(N)])  # ,2*np.ones(N),3*np.ones(N)])
y=y.astype(int)
thetas=np.array(#[[100,100,1,1]])
    [
        [1,1,1,1],
        [10,1,1,1],
        [10,10,1,1],
        [10,10,10,1],
    ]
)


xx, yy = make_meshgrid(X[:, 0], X[:, 1])
for theta, title, ax in zip(thetas,thetas, sub.flatten()):
    cls.fit(X, y,theta)
    plot_contours(ax, cls, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    # uni = cls.y
    # y1 = -(cls.w2.value[0, 0] * xx + cls.b2.value[0] - uni[0]) / cls.w2.value[0, 1]
    # ax.plot(xx, y1,color='black')
    #
    # y2 = -(cls.w2.value[1, 0] * xx + cls.b2.value[1] - uni[1]) / cls.w2.value[1, 1]
    # ax.plot(xx, y2,color='black')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    # ax.set_xlabel('X - axis')
    # ax.set_ylabel('Y - axis')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    #print("b",cls.b2.value)
    #print("w",cls.w2.value)

plt.show()




