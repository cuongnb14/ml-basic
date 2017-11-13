#  https://machinelearningcoban.com/2017/10/12/fundaml_vectors/#-gioi-thieu-ve-numpy

import numpy as np

help(np.array)

# Init array
# ------------------------------------------
x = np.array([1, 2, 3])
x = np.array([1, 2, 3], dtype=np.float64)

x = np.zeros(3)
x = np.ones(3)

x = np.array([1, 2, 3])
y = np.zeros_like(x)
y = np.ones_like(x)

np.arange(3)  # array([0, 1, 2])
np.arange(3, 6)  # array([3, 4, 5])
np.arange(0, 1, 0.1)  # array([ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9])
np.arange(5, 1, -0.9)  # array([ 5. ,  4.1,  3.2,  2.3,  1.4])


# dimension array
# ------------------------------------------
x = np.array([3, 4, 5])
print(x.shape) # (3,)

lenght = x.shape[0]


# Access array
# ------------------------------------------
a = np.arange(10)
ids = [1, 3, 4, 8]
print(a[ids])

a[-3:] # return last three elements
a[:3] # return first three elements

a[[1, 3, 5]] = 1 # <=> a[1] = a[3] = a[5] = 1
a[::-1] # reverse an array



def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

z = np.array([1,2,3])
print(softmax(z))
