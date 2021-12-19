import numpy as np
from config import params
from gradient_booster import GradientBooster
from sklearn import datasets, model_selection, metrics, tree

np.random.seed(42)
x = datasets.load_diabetes()['data']
y = datasets.load_diabetes()['target']
x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y)

def evaluate(model):
    print('Training score: ', metrics.r2_score(y_train, model.predict(x_train)),
          '\nTesting score:',metrics.r2_score(y_test, model.predict(x_test)))

m = model_selection.GridSearchCV(tree.DecisionTreeRegressor(), params)
m.fit(x_train, y_train)
print('Decision Tree results: ')
evaluate(m)

model = GradientBooster(20)
model.fit(x_train, y_train)
print('Gradient Booster results: ')
evaluate(model)
