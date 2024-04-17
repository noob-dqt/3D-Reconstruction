import json

import numpy as np
import cv2

"""
归一化八点算法计算基础矩阵（或者有内参的情况下计算本质矩阵，由Eflag控制，内参K可以不输入）
入口函数名为Run
输入为p1, p2即匹配的点(num of points, 2)，在有内参的情况下输入K，并且Eflag=True
返回一个3*3的基本矩阵或者本质矩阵
"""


class Fundamental_Matrix:
    @staticmethod
    def change2hom(coord):
        # 接受一个欧式坐标，并将坐标变为齐次坐标后返回
        # coord shape: (num_points, dim)
        if coord.ndim == 1:
            return np.hstack([coord, 1])
        return np.concatenate((coord, np.ones((coord.shape[0], 1))), axis=1)

    @staticmethod
    def normalize_points(points):
        """ Scale and translate image points so that centroid of the points
            are at the origin and avg distance to the origin is equal to sqrt(2).
        输入点集points（齐次坐标），形状(3 x n)
        返回: 归一化后的坐标以及使用的变换矩阵，即 p' = norm3d * p，p'是归一化坐标
        """
        x = points[0]
        y = points[1]
        center = points.mean(axis=1)  # 行均值
        cx = x - center[0]  # 中心
        cy = y - center[1]
        dist = np.sqrt(np.power(cx, 2) + np.power(cy, 2))
        scale = np.sqrt(2) / dist.mean()
        norm3d = np.array([
            [scale, 0, -scale * center[0]],
            [0, scale, -scale * center[1]],
            [0, 0, 1]
        ])

        return np.dot(norm3d, points), norm3d

    @staticmethod
    def calculate_F_E(x1, x2, Eflag=False):
        """ 使用归一化坐标计算F或者E
            系数矩阵A每行都是[x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1]
            返回求解出的F/E
        """
        p1x, p1y = x1[:2]
        p2x, p2y = x2[:2]
        A = np.array([
            p1x * p2x, p1x * p2y, p1x,
            p1y * p2x, p1y * p2y, p1y,
            p2x, p2y, np.ones(len(p1x))
        ]).T

        # SVD分解求最小二乘解
        U, S, V = np.linalg.svd(A)
        F = V[-1].reshape(3, 3)

        # F秩为2，最后一个奇异值置为0
        U, S, V = np.linalg.svd(F)
        S[-1] = 0
        if Eflag:
            S = [1, 1, 0]  # 秩为2且特征值相同
        F = np.dot(U, np.dot(np.diag(S), V))

        return F

    def Run(self, p1, p2, Eflag=False, K=None):
        """
        归一化八点算法计算基础矩阵（或者有内参的情况下计算本质矩阵，由Eflag控制，内参K可不输入）
        输入为p1, p2即匹配的点(num of points, 2)，返回一个3*3的基本矩阵或者本质矩阵
        """
        assert p1.shape[0] == p2.shape[0]
        # 变为齐次坐标
        p1 = self.change2hom(p1).T
        p2 = self.change2hom(p2).T
        if Eflag:
            assert K is not None
            p1 = np.dot(np.linalg.inv(K), p1)
            p2 = np.dot(np.linalg.inv(K), p2)
        # 归一化
        p1n, T1 = self.normalize_points(p1)
        p2n, T2 = self.normalize_points(p2)

        # 归一化坐标计算F或者E
        F = self.calculate_F_E(p1n, p2n, Eflag)

        # 归一化的逆过程
        F = np.dot(T1.T, np.dot(F, T2))
        # F = np.dot(T2.T, np.dot(F, T1))
        # return F
        return F / F[2, 2]


"""
计算投影矩阵，输入F或者E，返回投影矩阵
"""


class Projection_Matrix:
    def get(self, x):
        return np.array([
            [0, -x[2], x[1]],
            [x[2], 0, -x[0]],
            [-x[1], x[0], 0]
        ])

    def reconstruct_one_point(self, pt1, pt2, m1, m2):
        """
        用于三角化点判断投影矩阵是否合理
        """
        A = np.vstack([
            np.dot(self.get(pt1), m1),
            np.dot(self.get(pt2), m2)
        ])
        U, S, V = np.linalg.svd(A)
        P = np.ravel(V[-1, :4])

        return P / P[3]

    def Run(self, F, Eflag=False, p1=None, p2=None, K=None):
        """ 计算第二个相机投影矩阵（相机1已经假设为规范化相机，投影矩阵为[I 0]）
            P2 = [-[b^]* F  b]，b为F^T最小奇异值右奇异向量，||b||=1
            E分解返回四种可能的组合，F则是返回一种
            p1,p2用于筛选E分解四种投影阵正确的一种
        """
        if Eflag:
            U, S, V = np.linalg.svd(F)
            if np.linalg.det(np.dot(U, V)) < 0:
                V = -V
            W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            P2s = [np.vstack((np.dot(U, np.dot(W, V)).T, U[:, 2])).T,
                   np.vstack((np.dot(U, np.dot(W, V)).T, -U[:, 2])).T,
                   np.vstack((np.dot(U, np.dot(W.T, V)).T, U[:, 2])).T,
                   np.vstack((np.dot(U, np.dot(W.T, V)).T, -U[:, 2])).T]

            assert p1.shape[0] == p2.shape[0]
            # 变为齐次坐标
            p1 = Fundamental_Matrix.change2hom(p1).T
            p2 = Fundamental_Matrix.change2hom(p2).T
            assert K is not None
            points1n = np.dot(np.linalg.inv(K), p1)
            points2n = np.dot(np.linalg.inv(K), p2)
            P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])  # 第一个相机是[I 0]
            # 筛选投影矩阵
            idx = -1
            for i, P2 in enumerate(P2s):
                d1 = self.reconstruct_one_point(points1n[:, 0], points2n[:, 0], P1, P2)
                P2_homogenous = np.linalg.inv(np.vstack([P2, [0, 0, 0, 1]]))
                d2 = np.dot(P2_homogenous[:3, :4], d1)
                if d1[2] > 0 and d2[2] > 0:
                    idx = i
            return np.linalg.inv(np.vstack([P2s[idx], [0, 0, 0, 1]]))[:3, :4]
        else:
            U, S, V = np.linalg.svd(F.T)
            b = V[:, np.argmin(S)]
            b = b / np.linalg.norm(b)
            bx = np.asarray([
                [0, -b[2], b[1]],
                [b[2], 0, -b[0]],
                [-b[1], b[0], 0]
            ])
            A = np.matmul(-bx, F)
            # P2 = [A b]
            P2 = np.concatenate((A, np.expand_dims(b, axis=1)), axis=-1)
            return P2
            # return np.linalg.inv(np.vstack([P2, [0, 0, 0, 1]]))[:3, :4]


class Triangulation:
    def Run(self, p1, p2, m1, m2, Eflag=False, K=None):
        """
        p1 = m1 * X
        p2 = m2 * X， 求解 AX = 0.
        输入 p1, p2: 匹配的像素点，m1, m2投影矩阵 (3 x 4)
        返回 4 x n 3d点齐次坐标
        """
        assert p1.shape[0] == p2.shape[0]
        # 变为齐次坐标
        p1 = Fundamental_Matrix.change2hom(p1).T
        p2 = Fundamental_Matrix.change2hom(p2).T
        if Eflag:
            assert K is not None
            p1 = np.dot(np.linalg.inv(K), p1)
            p2 = np.dot(np.linalg.inv(K), p2)

        num_points = p1.shape[1]
        res = np.ones((4, num_points))

        for i in range(num_points):
            A = np.asarray([
                (p1[0, i] * m1[2, :] - m1[0, :]),
                (p1[1, i] * m1[2, :] - m1[1, :]),
                (p2[0, i] * m2[2, :] - m2[0, :]),
                (p2[1, i] * m2[2, :] - m2[1, :])
            ])
            _, _, V = np.linalg.svd(A)  # 线性法
            X = V[-1, :4]
            res[:, i] = X / X[3]

        return res

    def FRun(self, p1, p2, M1, M2):
        """
        p1 = m1 * X
        p2 = m2 * X，
        A = [um3 - m1,vm3 - m2,u'm3' - m1', v'm3'-m2']
        里面m1~m3对应m1的三个行向量，uv对应p1的xy，加上'对应m2，p2的相应内容
        求解 AX = 0，四个方程，求三个未知数，最小二乘解
        输入 p1, p2: 匹配的像素点，m1, m2投影矩阵 (3 x 4)
        返回 4 x n 3d点齐次坐标
        """
        assert p1.shape[0] == p2.shape[0]
        num_points = p1.shape[0]

        res = []
        for i in range(num_points):
            m1, m2, m3 = M1[0, :], M1[1, :], M1[2, :]
            m1_, m2_, m3_ = M2[0, :], M2[1, :], M2[2, :]
            u, v = p1[i]
            u_, v_ = p2[i]
            A = np.array([u * m3 - m1, v * m3 - m2, u_ * m3_ - m1_, v_ * m3_ - m2_])
            _, _, V = np.linalg.svd(A)  # 线性法
            X = V[:, -1]  # 最右解
            res.append(X / X[3])
        res = np.asarray(res).T
        return res


"""
用于标定相机获取内参，保存并返回内参
"""


class camera_calibration:
    def Run(self):
        # 棋盘格尺寸
        row = 10
        column = 7
        objpoint = np.zeros((row * column, 3), np.float32)
        objpoint[:, :2] = np.mgrid[0:row, 0:column].T.reshape(-1, 2)

        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        bimgs = ['images/b' + str(x) + '.jpg' for x in range(1, 6)]
        # 循环中断
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        imgGray = None
        for i, fname in enumerate(bimgs):
            img = cv2.imread(fname)
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # find chess board corners
            ret, corners = cv2.findChessboardCorners(imgGray, (row, column), None)
            # if found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objpoint)
                corners2 = cv2.cornerSubPix(imgGray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (row, column), corners2, ret)
                cv2.imwrite('tmp/' + str(i) + '.jpg', img)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imgGray.shape[::-1], None, None)
        res = {'ret': ret, 'mtx': mtx, 'dist': dist, 'rvecs': rvecs, 'tvecs': tvecs}
        path = 'cam_calib.txt'
        data = {
            "ret": str(ret),
            "mtx": mtx.tolist(),
            "dist": dist.tolist(),
            "rvecs": [rvec.tolist() for rvec in rvecs],
            "tvecs": [tvec.tolist() for tvec in tvecs]
        }
        json_data = json.dumps(data, indent=4)
        with open(path, 'w') as f:
            f.write(json_data)

        # meanError = 0
        # for i in range(len(objpoints)):
        #     imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        #     error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        #     meanError += error
        # print("total error: ", meanError / len(objpoints))
        return mtx

    def std(self):
        pass
