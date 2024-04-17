import numpy as np
import cv2
from scipy.optimize import curve_fit


def svd_cal(A, b=None):  # Ax=b，如果b=None，就是求Ax=0
    # 要求方程超定，列满秩，且方程个数多于未知数个数
    if b is not None:
        # 法二
        U, S, Vt = np.linalg.svd(A)
        y = (U.T @ b)[:S.size]
        y = y / S
        res2 = Vt.T @ y
        # 法三，库函数
        res3, residuals, rank, s = np.linalg.lstsq(A, b)  # 最小二乘解、残差平方和、系数矩阵的秩、系数矩阵的奇异值

        return res3
    else:
        U, S, Vt = np.linalg.svd(A)
        res1 = Vt.T[:, -1]
        return res1


def getNorm_mat(pts):
    # 归一化数据，输入N*2，返回使用的变换矩阵
    x, y = pts[:, 0], pts[:, 1]
    xm, ym, xv, yv = x.mean(), y.mean(), x.var(), y.var()
    s_x, s_y = np.sqrt(2. / xv), np.sqrt(2. / yv)
    norm_matrix = np.array([[s_x, 0., -s_x * xm],
                            [0., s_y, -s_y * ym],
                            [0., 0., 1.]])
    return norm_matrix


def f_refine(xdata, *params):
    # L-M方法的价值函数
    h11, h12, h13, h21, h22, h23, h31, h32, h33 = params
    N = xdata.shape[0] // 2
    X, Y = xdata[:N], xdata[N:]
    x = (h11 * X + h12 * Y + h13) / (h31 * X + h32 * Y + h33)
    y = (h21 * X + h22 * Y + h23) / (h31 * X + h32 * Y + h33)
    res = np.zeros_like(xdata)
    res[:N] = x
    res[N:] = y
    return res


# L-M方法的雅可比函数,curve_fit会自动算,可以不用这个参数
'''
def jac_refine(xdata, *params):
    h11, h12, h13, h21, h22, h23, h31, h32, h33 = params
    N = xdata.shape[0] // 2
    X, Y = xdata[:N], xdata[N:]
    J = np.zeros((N * 2, 9))
    J_x = J[:N]
    J_y = J[N:]
    s_x = h11 * X + h12 * Y + h13
    s_y = h21 * X + h22 * Y + h23
    w = h31 * X + h32 * Y + h33
    w_sq = w ** 2
    J_x[:, 0] = X / w
    J_x[:, 1] = Y / w
    J_x[:, 2] = 1. / w
    J_x[:, 6] = (-s_x * X) / w_sq
    J_x[:, 7] = (-s_x * Y) / w_sq
    J_x[:, 8] = -s_x / w_sq

    J_y[:, 3] = X / w
    J_y[:, 4] = Y / w
    J_y[:, 5] = 1. / w
    J_y[:, 6] = (-s_y * X) / w_sq
    J_y[:, 7] = (-s_y * Y) / w_sq
    J_y[:, 8] = -s_y / w_sq
    J[:N] = J_x
    J[N:] = J_y
    return J
'''


def cal_homography(p3d, p2d):
    # 算单应矩阵并且做非线性优化
    p3d_bak, p2d_bak = p3d.copy(), p2d.copy()
    # 最小二乘法计算单应性矩阵
    N = p3d.shape[0]
    # 归一化
    norm_mat_p3d = getNorm_mat(p3d)
    norm_mat_p2d = getNorm_mat(p2d)
    p3d = np.hstack((p3d, np.ones((N, 1))))  # 转齐次坐标
    p2d = np.hstack((p2d, np.ones((N, 1))))
    p3d_norm = np.dot(p3d, norm_mat_p3d.T)
    p2d_norm = np.dot(p2d, norm_mat_p2d.T)
    X, Y, x, y = p3d_norm[:, 0], p3d_norm[:, 1], p2d_norm[:, 0], p2d_norm[:, 1]
    # 构造一个系数矩阵
    A = np.zeros((N * 2, 9))
    xc = np.zeros((N, 9))
    xc[:, 0] = -X
    xc[:, 1] = -Y
    xc[:, 2] = -1.
    xc[:, 6] = x * X
    xc[:, 7] = x * Y
    xc[:, 8] = x
    A[:N] = xc
    yc = np.zeros((N, 9))
    yc[:, 3] = -X
    yc[:, 4] = -Y
    yc[:, 5] = -1.
    yc[:, 6] = y * X
    yc[:, 7] = y * Y
    yc[:, 8] = y
    A[N:] = yc
    H_norm = svd_cal(A).reshape((3, 3))  # svd分解求解H
    H = np.dot(np.dot(np.linalg.inv(norm_mat_p2d), H_norm), norm_mat_p3d)  # 去归一化
    # return H # 优化前的解
    # 对H非线性优化
    X, Y, x, y = p3d_bak[:, 0], p3d_bak[:, 1], p2d_bak[:, 0], p2d_bak[:, 1]
    N = X.shape[0]
    h0 = H.ravel()
    x_ = np.zeros(N * 2)
    x_[:N], x_[N:] = X, Y
    y_ = np.zeros(N * 2)
    y_[:N], y_[N:] = x, y
    # 用Levenberg-Marquardt算法优化H矩阵
    h_refined, _ = curve_fit(f_refine, x_, y_, p0=h0)
    h_refined /= h_refined[-1]
    H_refined = h_refined.reshape((3, 3))
    return H_refined


def get_vij(Hs, i, j):
    # 获取内参正交约束
    vij = np.zeros((Hs.shape[0], 6))
    vij[:, 0] = Hs[:, 0, i] * Hs[:, 0, j]
    vij[:, 1] = Hs[:, 0, i] * Hs[:, 1, j] + Hs[:, 1, i] * Hs[:, 0, j]
    vij[:, 2] = Hs[:, 1, i] * Hs[:, 1, j]
    vij[:, 3] = Hs[:, 2, i] * Hs[:, 0, j] + Hs[:, 0, i] * Hs[:, 2, j]
    vij[:, 4] = Hs[:, 2, i] * Hs[:, 1, j] + Hs[:, 1, i] * Hs[:, 2, j]
    vij[:, 5] = Hs[:, 2, i] * Hs[:, 2, j]
    return vij


def get_intrinsics(Hs):
    # H矩阵恢复内参,输入是一个H列表,H数量大于等于3,返回计算出的相机内参K
    N = len(Hs)
    Hs = np.stack(Hs)  # (N*3*3)
    v00, v01, v11 = get_vij(Hs, 0, 0), get_vij(Hs, 0, 1), get_vij(Hs, 1, 1)
    # 生成系数矩阵 2Nx6
    V = np.zeros((2 * N, 6))
    V[:N] = v01
    V[N:] = v00 - v11
    # SVD求解Vb = 0
    b = svd_cal(V)
    # B = K^-T K^-1
    B = np.array([[b[0], b[1], b[3]],
                  [b[1], b[2], b[4]],
                  [b[3], b[4], b[5]]])
    # 求解K
    v = (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2]) / (B[0, 0] * B[1, 1] - B[0, 1] * B[0, 1])
    Lambda = B[2, 2] - (B[0, 2] * B[0, 2] + v * (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2])) / B[0, 0]
    alpha = np.sqrt(Lambda / B[0, 0])
    beta = np.sqrt(Lambda * B[0, 0] / (B[0, 0] * B[1, 1] - B[0, 1] * B[0, 1]))
    gamma = -B[0, 1] * alpha * alpha * beta / Lambda
    u = gamma * v / beta - B[0, 2] * alpha * alpha / Lambda
    K = np.array([[alpha, gamma, u],
                  [0., beta, v],
                  [0., 0., 1.]])
    return K


def get_extrinsics(H, K):
    # 用H和K计算外参数矩阵,返回3X4的外参
    h0, h1, h2 = H[:, 0], H[:, 1], H[:, 2]
    K_inv = np.linalg.inv(K)
    Lambda = 1. / np.linalg.norm(np.dot(K_inv, h0))
    # 直接算r0,r1,t,r2叉乘获取
    r0, r1 = Lambda * np.dot(K_inv, h0), Lambda * np.dot(K_inv, h1)
    r2 = np.cross(r0, r1)
    t = Lambda * np.dot(K_inv, h2)
    R = np.vstack((r0, r1, r2)).T
    # R SVD分解,让其满足正交矩阵约束,即令S为I
    U, S, V_t = np.linalg.svd(R)
    R = np.dot(U, V_t)
    return np.hstack((R, t[:, np.newaxis]))  # 返回[R,t]


def get_distortion(p3d, p2ds, K, Rts):
    # 最小二乘求解畸变系数,输入N*2 3D坐标,M个N*2的2D坐标列表,M个外参的列表,返回径向畸变系数k0和k1
    M, N = len(p2ds), p3d.shape[0]
    p3d = np.hstack((p3d, np.zeros((p3d.shape[0], 1))))  # 补上Z的0坐标,并转化成齐次
    p3d = np.hstack((p3d, np.ones((p3d.shape[0], 1))))
    uc, vc = K[0, 2], K[1, 2]
    r = np.zeros(2 * M * N)
    for i, E in enumerate(Rts):
        proj_norm = np.dot(p3d, E.T)
        proj_norm /= proj_norm[:, -1].reshape((proj_norm.shape[0], 1))
        proj_norm = proj_norm[:, :-1]
        x_, y_ = proj_norm[:, 0], proj_norm[:, 1]
        r_i = np.sqrt(x_ ** 2 + y_ ** 2)
        r[i * N:(i + 1) * N] = r_i
    r[M * N:] = r[:M * N]
    # 观测向量
    obs = np.zeros(2 * M * N)
    u_o, v_o = np.zeros(M * N), np.zeros(M * N)
    for i, p2d in enumerate(p2ds):
        u_i, v_i = p2d[:, 0], p2d[:, 1]
        u_o[i * N:(i + 1) * N] = u_i
        v_o[i * N:(i + 1) * N] = v_i
    obs[:M * N] = u_o
    obs[M * N:] = v_o
    # 预测向量
    pred = np.zeros(2 * M * N)
    pred_ = np.zeros(2 * M * N)
    u_p, v_p = np.zeros(M * N), np.zeros(M * N)
    for i, E in enumerate(Rts):
        P = np.dot(K, E)
        projection = np.dot(p3d, P.T)
        projection /= projection[:, -1].reshape((projection.shape[0], 1))
        projection = projection[:, :-1]
        u_pi, v_pi = projection[:, 0], projection[:, 1]
        u_p[i * N:(i + 1) * N] = u_pi
        v_p[i * N:(i + 1) * N] = v_pi
    pred[:M * N] = u_p
    pred[M * N:] = v_p
    pred_[:M * N] = u_p - uc
    pred_[M * N:] = v_p - vc
    D = np.zeros((2 * M * N, 2))
    D[:, 0] = pred_ * r ** 2
    D[:, 1] = pred_ * r ** 4
    b = obs - pred
    D_inv = np.linalg.pinv(D)  # 伪逆
    return np.dot(D_inv, b)


def project(K, k, Rt, p3d):
    # 投影3d点集到2d,输入内参,畸变系数,外参,3D点
    # 返回N*2的2D坐标
    p3d_hom = np.hstack((p3d, np.zeros((p3d.shape[0], 1))))  # 补上Z的0坐标,并转化成齐次
    p3d_hom = np.hstack((p3d_hom, np.ones((p3d.shape[0], 1))))
    norm_proj = np.dot(p3d_hom, Rt.T)
    norm_proj /= norm_proj[:, -1].reshape((p3d.shape[0], 1))
    norm_proj = norm_proj[:, :-1]
    x, y = norm_proj[:, 0], norm_proj[:, 1]
    r = np.sqrt(x ** 2 + y ** 2)
    D = k[0] * r ** 2 + k[1] * r ** 4
    x_p = x * (1. + D)
    y_p = y * (1. + D)
    distorted_proj = np.hstack((x_p[:, None], y_p[:, None]))
    distorted_proj_hom = np.hstack((distorted_proj, np.ones((distorted_proj.shape[0], 1))))
    return np.dot(distorted_proj_hom, K[:-1].T)


def f_refine_krt(xdata, *params):
    # 优化KRt的价值函数
    if len(params) < 7 or len(params[7:]) % 6 != 0:
        raise ValueError('Check parameter vector encoding')
    if xdata.ndim != 1:
        raise ValueError('Check data vector encoding')
    M = len(params[7:]) // 6
    N = xdata.shape[0] // (2 * M)
    X = xdata[:N]
    Y = xdata[N:2 * N]
    pts = np.zeros((N, 2))
    pts[:, 0] = X
    pts[:, 1] = Y
    # 还原参数形状
    alpha, beta, gamma, u_c, v_c, k0, k1 = params[:7]
    K = np.array([[alpha, gamma, u_c], [0., beta, v_c], [0., 0., 1.]])
    k = np.array([k0, k1])
    Rts = []
    for i in range(7, len(params), 6):
        rho_x, rho_y, rho_z, t_x, t_y, t_z = params[i:i + 6]
        R = cv2.Rodrigues(np.array([rho_x, rho_y, rho_z]))[0]  # 旋转向量转旋转矩阵
        t = np.array([t_x, t_y, t_z])
        E = np.concatenate((R, t.reshape((3, 1))), axis=1)
        Rts.append(E)
    # 生成观测向量
    obs_x = np.zeros(N * M)
    obs_y = np.zeros(N * M)
    for e, Rt in enumerate(Rts):
        proj2d = project(K, k, Rt, pts)
        x, y = proj2d[:, 0], proj2d[:, 1]
        obs_x[e * N:(e + 1) * N] = x
        obs_y[e * N:(e + 1) * N] = y
    res = np.zeros(2 * N * M)
    res[:N * M] = obs_x
    res[N * M:] = obs_y
    return res


def refine_paras(p3d, p2ds, K, k, Rts):
    # 非线性优化所有参数,传入3D(N*2),2D坐标(M*N*2),内参K(3*3),畸变系数k(1*2),外参Rts(M*3*4)
    # 返回优化后的K,k,以及所有Rt
    M = len(p2ds)
    N = p3d.shape[0]
    # 所有参数展平成一维向量
    packed_params = []
    alpha, beta, gamma, u_c, v_c = K[0, 0], K[1, 1], K[0, 1], K[0, 2], K[1, 2]
    packed_params.extend([alpha, beta, gamma, u_c, v_c, k[0], k[1]])
    for E in Rts:
        t = E[:, 3]
        rodrigues = cv2.Rodrigues(E[:3, :3])[0].ravel()  # R转换成旋转向量表示
        packed_params.extend([rodrigues[0], rodrigues[1], rodrigues[2], t[0], t[1], t[2]])
    p0 = np.array(packed_params)
    # 展平p3d
    xdata = np.zeros(M * N * 2)
    xdata[:N] = p3d[:, 0]
    xdata[N:2 * N] = p3d[:, 1]
    # 展平p2d
    obs_x, obs_y = [], []
    for p2d in p2ds:
        x, y = p2d[:, 0], p2d[:, 1]
        obs_x.append(x)
        obs_y.append(y)
    obs_x = np.hstack(obs_x)
    obs_y = np.hstack(obs_y)
    ydata = np.hstack((obs_x, obs_y))
    # LM优化重投影误差
    popt, _ = curve_fit(f_refine_krt, xdata, ydata, p0)
    # 还原内外参和畸变系数
    alpha, beta, gamma, u_c, v_c, k0, k1 = popt[:7]
    K_refined = np.array([[alpha, gamma, u_c], [0., beta, v_c], [0., 0., 1.]])
    k_refined = np.array([k0, k1])
    Rts_refined = []
    for i in range(7, len(popt), 6):
        rho_x, rho_y, rho_z, t_x, t_y, t_z = popt[i:i + 6]
        R = cv2.Rodrigues(np.array([rho_x, rho_y, rho_z]))[0]
        t = np.array([t_x, t_y, t_z])
        E = np.concatenate((R, t.reshape((3, 1))), axis=1)
        Rts_refined.append(E)

    return K_refined, k_refined, Rts_refined


def zhangzyCalibrate(p3d, p2ds):
    # 张正友标定法，计算内外参、畸变系数
    # p3d：输入3d世界坐标（不用z）且所有图都用一个3d坐标，形状为N*2，N是标定板角点个数也即row*col
    # p2ds：检测出角点的所有图像对应的像素点坐标，list，每个元素形状N*2
    # 返回内参、畸变系数、所有外参
    Hs = []  # 所有单应矩阵
    for p2d in p2ds:  # 算单应性矩阵H，并做非线性优化
        H = cal_homography(p3d, p2d)
        Hs.append(H)
    K = get_intrinsics(Hs)  # 内参
    # 算外参
    extrinsic_matrices = []
    for h, H in enumerate(Hs):
        E = get_extrinsics(H, K)
        extrinsic_matrices.append(E)
        # 生成投影矩阵,把3d点投影回2d,算误差,这步非必须
        # P = np.dot(K, E)
        # p3d_hom = np.hstack((p3d, np.zeros((p3d.shape[0], 1))))  # 补上Z的0坐标,并转化成齐次
        # p3d_hom = np.hstack((p3d_hom, np.ones((p3d.shape[0], 1))))
        # predicted = np.dot(p3d_hom, P.T)
        # predicted = util.to_inhomogeneous(predicted)
        # data = p2ds[h]
        # nonlinear_sse_decomp = np.sum((predicted - data) ** 2)

    # 用内外参数计算径向畸变(张定友方法只考虑畸变模型里的径向畸变)
    k = get_distortion(p3d, p2ds, K, extrinsic_matrices)  # [k0,k1]
    # 利用初始解进行非线性优化重投影误差
    K, k, Rts = refine_paras(p3d, p2ds, K, k, extrinsic_matrices)
    return K, k, Rts


def main():
    src = '../imgs/'  # 多张标定用的标定板图像目录(以1.jpg,2.jpg命名)
    row, col = 10, 7  # 内点数目,棋盘格数目-1(imgs下的是11*8大小,减1后在这里是10*7)
    p3d = np.mgrid[0:row, 0:col].T.reshape(-1, 2)
    p2ds = []  # 2d points
    bimgs = [src + str(x) + '.jpg' for x in range(1, 6)]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for i, fname in enumerate(bimgs):
        imgGray = cv2.imread(fname, 0)
        # 找角点
        fg, corners = cv2.findChessboardCorners(imgGray, (row, col), None)
        if fg:
            corner = cv2.cornerSubPix(imgGray, corners, (11, 11), (-1, -1), criteria)
            p2ds.append(corner.reshape(corner.shape[0], corner.shape[2]))
            img = cv2.drawChessboardCorners(imgGray, (row, col), corner, fg)
            cv2.imwrite('tmp/' + str(i) + '.jpg', img)
    K, k, Rts = zhangzyCalibrate(p3d, p2ds)

    print('内参:', K)
    print('畸变:k0={:.6f}  k1={:.6f}'.format(k[0], k[1]))
    for i, Rt in enumerate(Rts):
        print('外参数矩阵(旋转向量)' + str(i + 1) + ':', cv2.Rodrigues(Rt[:3, :3])[0].ravel())


if __name__ == '__main__':
    main()
