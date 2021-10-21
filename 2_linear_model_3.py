# https://scikit-learn.org/stable/modules/linear_model.html#setting-the-regularization-parameter-leave-one-out-cross-validation

import numpy as np
from sklearn import linear_model
reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
print(reg.alpha_)