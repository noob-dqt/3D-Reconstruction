import numpy as np
import cv2


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
        return res3  # 写的时候可以写自己实现的res2
    else:
        U, S, Vt = np.linalg.svd(A)
        res1 = Vt.T[:, -1]
        return res1


def getProjMat(P):  # 输入系数矩阵P，计算投影矩阵m
    m = cal(P)
    M = np.zeros((3, 4))
    M[0, :] = m[:4]
    M[1, :] = m[4:8]
    M[2, :] = m[8:]
    return M


# Pm = 0求解投影矩阵m，P为2n*12的系数矩阵
def getP(pts):  # 输入n*5的数组，5表示匹配的3D坐标x,y,z以及对应像素u,v
    n, _ = pts.shape
    P = np.zeros((2 * n, 12))
    for i, pi in enumerate(pts):
        # x, y, z, u, v = pi
        j = i * 2
        P[j, :3] = pi[:3]
        P[j, 3] = 1
        P[j, 8:11] = -pi[3] * pi[:3]
        P[j, 11] = -pi[3]
        #
        P[j + 1, 4:7] = pi[:3]
        P[j + 1, 7] = 1
        P[j + 1, 8:11] = -pi[4] * pi[:3]
        P[j + 1, 11] = -pi[4]
    return P


def get_KRT(M):  # 输入投影矩阵，QR分解获取K(3*3)、R(3*3)、T(3*1)
    Q = M[:, :3]
    b = M[:, -1]
    b = b.reshape(3, 1)
    Qinv = np.linalg.inv(Q)
    Rt, Kinv = np.linalg.qr(Qinv)
    K = np.linalg.inv(Kinv)
    K = abs(K / K[2, 2])
    R = Rt.T
    t = np.dot((np.linalg.inv(-Q)), b)
    # t = Kinv @ b
    return K, R, t


# R和t的计算结果都和上面的算法不一样，测试时用上面，写在试卷可以用这个（因为上面的R和t可视化后感觉比较合理的）
def testKRt(M):
    a1T, a2T, a3T = M[0, :-1], M[1, :-1], M[2, :-1]
    # p = 1 / np.linalg.norm(a3T)
    p = -1 / np.linalg.norm(a3T)

    u = p * p * np.dot(a1T, a3T)
    v = p * p * np.dot(a2T, a3T)
    r3 = a3T * p
    c13 = np.cross(a1T, a3T)
    c23 = np.cross(a2T, a3T)
    cos = - np.dot(c13, c23) / np.linalg.norm(c13) / np.linalg.norm(c23)
    # theta = np.arccos(cos)
    theta = np.pi / 2
    alpha = p * p * np.sin(theta) * np.linalg.norm(c13)
    belta = p * p * np.sin(theta) * np.linalg.norm(c23)
    r1 = c23 / np.linalg.norm(c23)
    r2 = np.cross(r3, r1)
    K = np.zeros((3, 3))
    K[0, 0] = alpha
    K[1, 1] = belta / np.sin(theta)
    K[2, 2] = 1
    K[0, 1] = -alpha / np.tan(theta)
    K[0, 2] = u
    K[1, 2] = v
    b = M[:, -1].reshape(3, 1)
    t = p * np.linalg.inv(K) @ b
    R = np.concatenate([r1.reshape(3, 1), r2.reshape(3, 1), r3.reshape(3, 1)], axis=1)
    return K, R, t


def main():
    # 点的数据从txt读取，里面有n（大于等于6）行，表示n组对应点，每行5个数据，空格隔开，前三个对应3D坐标xyz后两个为对应二维像素的坐标uv
    rs = np.loadtxt('./pts.txt')
    P = getP(rs)  # 系数矩阵
    M = getProjMat(P)
    np.set_printoptions(suppress=True)
    M /= M[-1, -1]
    print(M)
    # return
    # K, R, t = get_KRT(M)
    K, R, t = testKRt(M)
    print(K, R, t)
    # print(K @ np.concatenate((R, t.reshape((3, 1))), axis=1))
    # print(kk @ np.concatenate((rr, tt.reshape((3, 1))), axis=1))
    # 计算一下分解后K、R、t恢复的投影矩阵与原投影矩阵的差别
    # mm = K @ np.concatenate((R, t.reshape((3, 1))), axis=1)
    # print(np.linalg.norm(mm - M))


if __name__ == '__main__':
    main()
