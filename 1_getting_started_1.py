# from https://scikit-learn.org/stable/getting_started.html
# Fitting and predicting: estimator basics

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=0)

X = [[1, 2, 3],
     [11, 12, 13]]

y = [0, 1]
clf.fit(X, y)
aa = clf.predict(X)
print(aa)
xx = clf.predict([[4, 5, 6], [14, 15, 16]])
print(xx)