# https://scikit-learn.org/stable/getting_started.html#model-evaluation
# model evaluation

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

# make_regression 生成一个回归问题的数据
X, y = make_regression(n_samples=1000, random_state=0)
lr = LinearRegression()

result = cross_validate(lr, X, y)
print(result['test_score'])