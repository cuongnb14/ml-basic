#  https://machinelearningcoban.com/2017/10/20/fundaml_matrices/#-mang-nhieu-chieu

import numpy as np

# Init
# ------------------------------------------
A = np.array([[1, 2], [3, 4]])

# Ma trận đơn vị
# ------------------------------------------
np.eye(3)
np.eye(3, k=1)  # Ma tran don vi voi duong cheo phu nao do

# Ma trận đường chéo
# ------------------------------------------
np.diag([1, 3, 4])

np.diag(np.diag([1, 3, 4])) # Lay duong cheo cua ma tran

# dimension matrix
# ------------------------------------------
A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
A.shape # (3,4)


# Access matrix
# -----------------------------
A[1][2] == A[1, 2]

# Truy cập vào hàng/cột
A[0,:] == A[0]


# -----------
A = np.array([[1, 3], [2, 5]])

def norm_fro(matrix):
    return np.sqrt(np.sum(matrix ** 2))

print(norm_fro(A))


def fn(a):
    return np.sqrt(np.trace(a.dot(a.T)))

print(fn(A))