"""
步骤：
    1.对一个3D场景拍摄两张不同角度图像
    2.图像求特征点
    3.特征点匹配获取成对的像素点坐标（对应3D同一坐标）
    4.求出基本矩阵F（归一化八点法）
    5.由F求投影矩阵
    6.三角测量求出3D点坐标（射影重构的结果）
    7.标定相机获取内参矩阵
    8.进一步求本质矩阵E
    9.对E 分解出R、T
    10.三角测量求解坐标
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import orb_match as orb
import calculate_utils

if __name__ == '__main__':
    src1 = 'images/5.jpg'
    src2 = 'images/6.jpg'
    img1 = cv2.imread(src1)
    img2 = cv2.imread(src2)
    p1, p2 = orb.ORB().Run(img1, img2)
    # p1, p2 = orb.standard(img1, img2)

    '''
    p1 = np.int32(p1)
    p2 = np.int32(p2)
    F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_LMEDS)
    print('Fundamental matrix:\n', F)
    # p1 = pts1[mask.ravel() == 1]
    # p2 = pts2[mask.ravel() == 1]
    '''

    K = calculate_utils.camera_calibration().Run()
    F = calculate_utils.Fundamental_Matrix().Run(p1, p2)
    E = calculate_utils.Fundamental_Matrix().Run(p1, p2, True, K)
    # E = np.matmul(np.matmul(K.T, F), K)
    # print(F)
    print('Fundamental matrix:\n', F)
    print('Essential matrix:\n', E)
    # 计算投影矩阵
    P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])  # 第一个相机是[I 0]
    P2 = calculate_utils.Projection_Matrix().Run(F)  # 计算第二个相机投影阵
    # P2 = calculate_utils.Projection_Matrix().Run(E, True, p1, p2, K)
    print("P1:\n", P1)
    print("P2:\n", P2)
    # 三角化求3D点
    # Points3D = calculate_utils.Triangulation().Run(p1, p2, P1, P2)
    Points3D = calculate_utils.Triangulation().FRun(p1, p2, P1, P2)

    # 转换为欧式坐标 (3*n)
    points3d = Points3D[:3, :]
    min_value = np.min(points3d, axis=1)
    max_value = np.max(points3d, axis=1)
    # 归一化坐标
    points3d = (points3d - min_value[:, np.newaxis]) / (max_value - min_value)[:, np.newaxis]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 绘制点
    ax.scatter(points3d[0], points3d[1], points3d[2], c='red', marker='o')
    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
