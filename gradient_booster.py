import numpy as np
from sklearn import tree

class GradientBooster:
    def __init__(self, n_trees = 20):
        self.f = []
        self.learning_rate = []
        self.n_trees = n_trees

    def fit(self,x,y, lr=0.1):
        class F0:
            predict = lambda x:np.mean(y) * np.ones(x.shape[0])
        self.f.append(F0)
        self.learning_rate.append(1)

        for _ in range(self.n_trees):
            m = tree.DecisionTreeRegressor(max_depth=5)
            res = y - self.predict(x)
            m.fit(x, res)
            self.f.append(m)
            self.learning_rate.append(lr)

    def predict(self,x):
        return sum(f.predict(x) * lr for f, lr in zip(self.f, self.learning_rate))