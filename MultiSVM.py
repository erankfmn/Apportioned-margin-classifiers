import numpy as np
from cvxpy import *
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy import sparse as sp
from sklearn.svm.base import BaseSVC
from sklearn.utils.validation import indexable


class MultiSVM(BaseEstimator, ClassifierMixin):
    """MultiSvM class"""

    def linearKernel(self,x1, x2):
        return x1.dot(x2.T)  # /(np.linalg.norm(x1)*np.linalg.norm(x2))

    def RBFKernel(self,x1, x2, gamma=1):
        return np.exp(-np.linalg.norm((x2 - x1)) / self.gamma)

    def polyKernel(self,x1, x2, degree=2):
        return (1 + x1.dot(x2.T)) ** self.degree

    def __init__(self, class_weight=None, kernel='linear',degree =3,gamma=1,C=1):

        if isinstance(class_weight,dict):
            self.class_weight = np.array(list(class_weight.values()))
        else :
            self.class_weight=class_weight

        self.kernel=kernel
        if (kernel == 'linear'):
            self.ker = self.linearKernel
        if (kernel == 'rbf'):
            self.ker = self.RBFKernel
        if (kernel == 'poly'):
            self.ker = self.polyKernel

        self.degree=degree
        self.gamma=gamma
        self.C=C

    def fit1(self, X, y, sample_weight=None):
        """
          An attempt to solve the dual using SGD
        """
        # add intercept
        temp = np.ones([X.shape[0], 1])
        X = np.hstack((X, temp))
        self.y = np.unique(y)
        self.K = len(np.unique(y))
        self.alpha = np.zeros([X.shape[0],self.K])
        if (sample_weight is not None):
            self.class_weight = sample_weight


        for t in range(X.shape[0]*100):
            index = t%X.shape[0]#np.random.randint(0, X.shape[0])#
            yt=y[index]
            for j in self.y:
                if (yt == j):
                    delta = 1
                else :
                    delta = -1
                sum=0
                for i in range(X.shape[0]):
                    sum += self.alpha[i,j] * self.ker(X[i, :], X[index, :])

                if (self.class_weight[yt] >= delta *sum):#((1/(lamda *(t+1)))*sum)):
                    self.alpha[index,j] += delta

        self.w2 = (X.T.dot(self.alpha)).T
        self.sv = X
    #    print("alphas are:", self.alpha)
        return self

    def fit2(self, X, y, sample_weight=None):
        self.y = np.unique(y)
        self.K = len(np.unique(y))
        self.w2 = Variable(self.K, X.shape[1])
        self.b = Variable(self.K)

        if(sample_weight is not None):
            self.class_weight=sample_weight

        obj = 0
        # for all points
        for i in range(X.shape[0]):
            # for all classes
            for j in range(self.K):
                if y[i] == self.y[j]:
                    obj += pos(self.class_weight[y[i]] - (self.w2[j, :] *( X[i, :]).T + self.b[j]))
                else:
                    obj += pos(self.class_weight[y[i]] + (self.w2[j, :]*( X[i, :]).T+ self.b[j]))

        reg = norm(self.w2)
        Problem(Minimize(self.C*obj + reg)).solve(solver=SCS)
        return self

    def predict1(self, X):
        temp = np.ones([X.shape[0], 1])
        X = np.hstack((X, temp))
        y = np.empty(X.shape[0])
        for i in range((X.shape[0])):
            sum = np.zeros(self.K)
            for j in range(self.K):
                for k in range(self.sv.shape[0]):
                    sum[j] += self.alpha[k,j] * self.ker(self.sv[k], X[i, :])
                sum[j]= (sum[j])/self.class_weight[j]
            y[i] = self.y[sum.argmax()]
        return y

    def predict2(self, X):
        y = np.empty(X.shape[0])
        for i in range((X.shape[0])):
            r = (X[i, :] * self.w2.value.T+ self.b.value.T)/self.class_weight
            y[i] = self.y[r.argmax()]
        return y

    fit = fit1
    predict = predict1

