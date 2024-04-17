import numpy as np
import cv2
import json


def std():  # 标定板图像放在images里
    # 棋盘格尺寸，角点数（11，8）-> 10,7
    row = 10
    column = 7
    objpoint = np.zeros((row * column, 3), np.float32)
    objpoint[:, :2] = np.mgrid[0:row, 0:column].T.reshape(-1, 2)

    objpoints = []  # 3d point
    imgpoints = []  # 2d points
    bimgs = ['../imgs/b' + str(x) + '.jpg' for x in range(1, 6)]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # 循环中断条件
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
    Rs = [cv2.Rodrigues(vec)[0] for vec in rvecs] # 旋转向量转化为旋转矩阵
    path = '../cam_calib.txt'
    data = {
        "ret": str(ret),  #
        "mtx": mtx.tolist(),  # 内参
        "dist": dist.tolist(),  # 畸变系数，distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
        "R": [r.tolist() for r in Rs],  # 旋转
        "rvecs": [rvec.tolist() for rvec in rvecs],  # 旋转
        "tvecs": [tvec.tolist() for tvec in tvecs]  # 平移向量
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


std()
