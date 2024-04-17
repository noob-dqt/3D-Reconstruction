import numpy as np


def cal(A, b=None):  # Ax=b，如果b=None，就是求Ax=0
    # 要求方程超定，列满秩，且方程个数多于未知数个数
    if b is not None:
        # 法一
        # res1 = np.linalg.inv(A.T @ A) @ A.T @ b # A.T@A不一定是可逆的
        # 法二
        U, S, Vt = np.linalg.svd(A)
        y = (U.T @ b)[:S.size]
        y = y / S
        res2 = Vt.T @ y

        # 法三，库函数
        res3, residuals, rank, s = np.linalg.lstsq(A, b)  # 最小二乘解、残差平方和、系数矩阵的秩、系数矩阵的奇异值

        return res2, res3
    else:
        U, S, Vt = np.linalg.svd(A)
        res1 = Vt.T[:, -1]
        return res1


# A = np.random.rand(5, 3)
# b = np.random.rand(5)
# print(cal(A, b)) # 非齐次
# print(cal(A)) # 齐次
