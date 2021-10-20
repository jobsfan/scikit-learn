# https://scikit-learn.org/stable/getting_started.html#transformers-and-pre-processors
# Transformers and pre-processors

from sklearn.preprocessing import StandardScaler

X = [[0, 15],
     [1, -10]]

r = StandardScaler().fit(X).transform(X)
print(r)