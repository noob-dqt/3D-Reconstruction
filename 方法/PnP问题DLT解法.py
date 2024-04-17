import numpy as np


def cal(A, b=None):  # Ax=b，如果b=None，就是求Ax=0
    # 要求方程超定，列满秩，且方程个数多于未知数个数
    if b is not None:
        # 法二
        U, S, Vt = np.linalg.svd(A)
        y = (U.T @ b)[:S.size]
        y = y / S
        res2 = Vt.T @ y
        # 法三，库函数
        res3, residuals, rank, s = np.linalg.lstsq(A, b)  # 最小二乘解、残差平方和、系数矩阵的秩、系数矩阵的奇异值
        return res3  # 写的时候可以写自己实现的res2
    else:
        U, S, Vt = np.linalg.svd(A)
        res1 = Vt.T[:, -1]
        return res1


def solve(pts, K):
    """
    输入Nx5的点集，一行对应一个匹配点，前五个对应为3D点坐标，后两个对应2D坐标，n>=6
    输入一个内参K，大小为3*3
    返回3*4的矩阵[R t]
    lambda * p = K * [R t] P，K已知，p，P已知，[R t]共12个约束
    """
    N = pts.shape[0]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    A = np.zeros((2 * N, 12))
    for i in range(N):
        j = 2 * i
        x, y, z, u, v = pts[i]
        A[j, :4] = x * fx, y * fy, z * fx, fx
        A[j, 8:] = x * cx - u * x, y * cx - u * y, z * cx - u * z, cx - u

        A[j + 1, 4:8] = x * fy, y * fy, z * fy, fy
        A[j + 1, 8:] = x * cy - v * x, y * cy - v * y, z * cy - v * z, cy - v
    res = cal(A)
    res = res.reshape(3, 4)
    R, t = res[:, :3], res[:, -1]
    # 近似一个符合约束的R
    U, S, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) <= 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    return R, t


def solve2(pts, K):
    # 来源于https://zhuanlan.zhihu.com/p/408703054的另一种推导过程，与PPT有一定区别
    N = pts.shape[0]
    A = np.zeros((2 * N, 12))
    for i in range(N):
        j = 2 * i
        x, y, z, u, v = pts[i]
        A[j, :4] = x, y, z, 1
        A[j, 8:] = - u * x, - u * y, - u * z, - u
        A[j + 1, 4:8] = x, y, z, 1
        A[j + 1, 8:] = - v * x, - v * y, - v * z, - v
    P = cal(A).reshape(3, 4)
    Kinv = np.linalg.inv(K)
    U, S, Vt = np.linalg.svd(Kinv @ P[:, :-1])
    R = U @ Vt
    t = np.dot(Kinv, P[:, -1]) / S[0]
    return R, t  # 算出来的t和三位标定物DLT testKRt算出来的t接近，但是R又不同


# pt = np.random.randn(7, 5)
pt = np.loadtxt('三维标定物/data/img_01.txt')
K_ = np.array(
    [[833.79616235, 16.26902594, 576.82528312],
     [0., 869.68482155, 852.67792169],
     [0., 0., 1.]]
)

r1, t1 = solve(pt, K_)
r2, t2 = solve2(pt, K_)
print(np.linalg.norm(r1 - r2), np.linalg.norm(t1 - t2))
