#  Ref: https://machinelearningcoban.com/2016/12/28/linearregression/
# ----------------------------------------------------------------------

# To support both python 2 and python 3
# from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import numpy

# height (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
y = np.array([[49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
# Test: 155cm - 52kg; 160cm - 56kg


# Building Xbar
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)

# Calculating weights of the fitting line
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)

w = np.dot(np.linalg.pinv(A), b)
# Preparing the fitting line
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(145, 185, 2)
y0 = w_0 + w_1 * x0

print("52 -  " + str(w_0 + w_1 * 155))
print("56 -  " + str(w_0 + w_1 * 160))


# plt.show()

# --------------------------------------------
from sklearn import linear_model

regr = linear_model.LinearRegression(fit_intercept=False)  # fit_intercept = False for calculating the bias
regr.fit(Xbar, y)

w_01 = regr.coef_[0][0]
w_11 = regr.coef_[0][1]

x1 = np.linspace(145, 185, 2)
y1 = w_01 + w_11 * x1


# Compare two results
print('Solution found by scikit-learn: ', regr.coef_)
print('Solution found by calculate formula: ', w.T)


# Drawing the fitting line
plt.plot(X.T, y.T, 'ro')  # data
# plt.plot(x0, y0)  # the fitting line
plt.plot(x1, y1)  # the fitting line

plt.axis([140, 190, 45, 75])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()
