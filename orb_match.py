import cv2
import matplotlib.pyplot as plt
from skimage.feature import plot_matches
import numpy as np
from scipy.signal import convolve2d
from scipy.spatial.distance import cdist

"""
完成特征检测和匹配，返回成对像素点
使用ORB特征检测算法进行特征检测和匹配，返回匹配的结果，以numpy数组返回
形状为[match_num,2]的两个float数组points1和points2，表示两张图共match_num个匹配，2表示像素坐标
入口为成员函数Run()，接受两个参数，分别是两视角的图像
"""


class ORB:
    @staticmethod
    def FAST(self, img, N=9, threshold=0.15, nms_window=2):
        kernel = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]]) / 16  # 3x3 Gaussian Window

        img = convolve2d(img, kernel, mode='same')

        cross_idx = np.array([[3, 0, -3, 0], [0, 3, 0, -3]])
        circle_idx = np.array([[3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3],
                               [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1]])

        corner_img = np.zeros(img.shape)
        keypoints = []
        for y in range(3, img.shape[0] - 3):
            for x in range(3, img.shape[1] - 3):
                It = img[y, x]
                t = threshold * It if threshold < 1 else threshold
                if np.count_nonzero(It + t < img[y + cross_idx[0, :], x + cross_idx[1, :]]) >= 3 or np.count_nonzero(
                        It - t > img[y + cross_idx[0, :], x + cross_idx[1, :]]) >= 3:
                    if np.count_nonzero(img[y + circle_idx[0, :], x + circle_idx[1, :]] >= It + t) >= N \
                            or np.count_nonzero(img[y + circle_idx[0, :], x + circle_idx[1, :]] <= It - t) >= N:
                        # KP
                        keypoints.append([x, y])  # kp = [col, row]
                        corner_img[y, x] = np.sum(np.abs(It - img[y + circle_idx[0, :], x + circle_idx[1, :]]))

        # NMS
        if nms_window != 0:
            fewer_kps = []
            for [x, y] in keypoints:
                window = corner_img[y - nms_window:y + nms_window + 1, x - nms_window:x + nms_window + 1]
                loc_y_x = np.unravel_index(window.argmax(), window.shape)
                x_new = x + loc_y_x[1] - nms_windowb
                y_new = y + loc_y_x[0] - nms_window
                new_kp = [x_new, y_new]
                if new_kp not in fewer_kps:
                    fewer_kps.append(new_kp)
        else:
            fewer_kps = keypoints

        return np.array(fewer_kps)

    @staticmethod
    def corner_orientations(self, img, corners):
        OFAST_MASK = np.zeros((31, 31), dtype=np.int32)
        OFAST_UMAX = [15, 15, 15, 15, 14, 14, 14, 13, 13, 12, 11, 10, 9, 8, 6, 3]
        for i in range(-15, 16):
            for j in range(-OFAST_UMAX[abs(i)], OFAST_UMAX[abs(i)] + 1):
                OFAST_MASK[15 + j, 15 + i] = 1
        mrows, mcols = OFAST_MASK.shape
        mrows2 = int((mrows - 1) / 2)
        mcols2 = int((mcols - 1) / 2)

        # Padding 0
        img = np.pad(img, (mrows2, mcols2), mode='constant', constant_values=0)

        # 强度质心法计算orientation
        orientations = []
        for i in range(corners.shape[0]):
            c0, r0 = corners[i, :]
            m01, m10 = 0, 0
            for r in range(mrows):
                m01_temp = 0
                for c in range(mcols):
                    if OFAST_MASK[r, c]:
                        Img = img[r0 + r, c0 + c]
                        m10 = m10 + Img * (c - mcols2)
                        m01_temp = m01_temp + Img
                m01 = m01 + m01_temp * (r - mrows2)
            orientations.append(np.arctan2(m01, m10))

        return np.array(orientations)

    @staticmethod
    def BRIEF(self, img, keypoints, orientations=None, n=256, patch_size=9, sigma=1, mode='uniform', sample_seed=42):
        random = np.random.RandomState(seed=sample_seed)
        kernel = np.array([[1, 4, 7, 4, 1],
                           [4, 16, 26, 16, 4],
                           [7, 26, 41, 26, 7],
                           [4, 16, 26, 16, 4],
                           [1, 4, 7, 4, 1]]) / 273  # 5x5 Gaussian Window
        img = convolve2d(img, kernel, mode='same')
        pos1 = pos2 = None
        if mode == 'normal':
            samples = (patch_size / 5.0) * random.randn(n * 8)
            samples = np.array(samples, dtype=np.int32)
            samples = samples[(samples < (patch_size // 2)) & (samples > - (patch_size - 2) // 2)]
            pos1 = samples[:n * 2].reshape(n, 2)
            pos2 = samples[n * 2:n * 4].reshape(n, 2)
        elif mode == 'uniform':
            samples = random.randint(-(patch_size - 2) // 2 + 1, (patch_size // 2), (n * 2, 2))
            samples = np.array(samples, dtype=np.int32)
            pos1, pos2 = np.split(samples, 2)
        rows, cols = img.shape
        if orientations is None:
            mask = (((patch_size // 2 - 1) < keypoints[:, 0])
                    & (keypoints[:, 0] < (cols - patch_size // 2 + 1))
                    & ((patch_size // 2 - 1) < keypoints[:, 1])
                    & (keypoints[:, 1] < (rows - patch_size // 2 + 1)))

            keypoints = np.array(keypoints[mask, :], dtype=np.intp, copy=False)
            descriptors = np.zeros((keypoints.shape[0], n), dtype=bool)

            for p in range(pos1.shape[0]):
                pr0 = pos1[p, 0]
                pc0 = pos1[p, 1]
                pr1 = pos2[p, 0]
                pc1 = pos2[p, 1]
                for k in range(keypoints.shape[0]):
                    kr = keypoints[k, 1]
                    kc = keypoints[k, 0]
                    if img[kr + pr0, kc + pc0] < img[kr + pr1, kc + pc1]:
                        descriptors[k, p] = True
        else:
            distance = int((patch_size // 2) * 1.5)
            mask = (((distance - 1) < keypoints[:, 0])
                    & (keypoints[:, 0] < (cols - distance + 1))
                    & ((distance - 1) < keypoints[:, 1])
                    & (keypoints[:, 1] < (rows - distance + 1)))

            keypoints = np.array(keypoints[mask], dtype=np.intp, copy=False)
            orientations = np.array(orientations[mask], copy=False)
            descriptors = np.zeros((keypoints.shape[0], n), dtype=bool)

            for i in range(descriptors.shape[0]):
                angle = orientations[i]
                sin_theta = np.sin(angle)
                cos_theta = np.cos(angle)

                kr = keypoints[i, 1]
                kc = keypoints[i, 0]
                for p in range(pos1.shape[0]):
                    pr0 = pos1[p, 0]
                    pc0 = pos1[p, 1]
                    pr1 = pos2[p, 0]
                    pc1 = pos2[p, 1]

                    # x` = x*cos(th) - y*sin(th)
                    # y` = x*sin(th) + y*cos(th)
                    # c -> x & r -> y
                    spr0 = round(sin_theta * pr0 + cos_theta * pc0)
                    spc0 = round(cos_theta * pr0 - sin_theta * pc0)
                    spr1 = round(sin_theta * pr1 + cos_theta * pc1)
                    spc1 = round(cos_theta * pr1 - sin_theta * pc1)

                    if img[kr + spr0, kc + spc0] < img[kr + spr1, kc + spc1]:
                        descriptors[i, p] = True
        return descriptors

    @staticmethod
    def match(self, descriptors1, descriptors2, max_distance=np.inf, cross_check=True, distance_ratio=None):

        distances = cdist(descriptors1, descriptors2, metric='hamming')  # distances.shape: [len(d1), len(d2)]

        indices1 = np.arange(descriptors1.shape[0])  # [0, 1, 2, 3, 4, 5, 6, 7, ..., len(d1)]
        indices2 = np.argmin(distances, axis=1)  # [12, 465, 23, ..., len(d1)]，最接近d1 points的d2 points idx
        # 对于d1 point都有一个d2 point与之最接近
        if cross_check:
            matches1 = np.argmin(distances, axis=0)  # [15, 37, 283, ..., len(d2)]，最接近的d2 points的d1 points
            mask = indices1 == matches1[indices2]  # len(mask) = len(d1)
            indices1 = indices1[mask]
            indices2 = indices2[mask]

        if max_distance < np.inf:
            mask = distances[indices1, indices2] < max_distance
            indices1 = indices1[mask]
            indices2 = indices2[mask]

        if distance_ratio is not None:
            # 筛选误匹配
            # 如果最近的匹配距离与第二近的匹配距离之间的比值超过定义的ratio，则删除该匹配。
            modified_dist = distances
            fc = np.min(modified_dist[indices1, :], axis=1)
            modified_dist[indices1, indices2] = np.inf
            fs = np.min(modified_dist[indices1, :], axis=1)
            mask = fc / fs <= distance_ratio
            indices1 = indices1[mask]
            indices2 = indices2[mask]

        # 根据距离对匹配排序
        dist = distances[indices1, indices2]
        sorted_indices = dist.argsort()
        _matches = np.column_stack((indices1[sorted_indices], indices2[sorted_indices]))
        return _matches

    def Run(self, img1, img2):  # 接收两个图像，运算后返回图像匹配像素坐标，x1,y1以及其对应的x2,y2
        original_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        original_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        features_img1 = np.copy(original_img1)
        features_img2 = np.copy(original_img2)
        kp1 = self.FAST(self, gray1, N=9, threshold=0.15, nms_window=3)  # 特征坐标
        kp2 = self.FAST(self, gray2, N=9, threshold=0.15, nms_window=3)
        for keypoint in kp1:
            features_img1 = cv2.circle(features_img1, tuple(keypoint), 3, (0, 0, 255), 1)
        for keypoint in kp2:
            features_img2 = cv2.circle(features_img2, tuple(keypoint), 3, (0, 0, 255), 1)

        # 绘制特征检测结果
        # plt.figure()
        # plt.suptitle('View left')
        # plt.subplot(1, 2, 1)
        # plt.imshow(gray1, cmap='gray')
        # plt.subplot(1, 2, 2)
        # plt.imshow(features_img1)
        #
        # plt.figure()
        # plt.suptitle('view right')
        # plt.subplot(1, 2, 1)
        # plt.imshow(gray2, cmap='gray')
        # plt.subplot(1, 2, 2)
        # plt.imshow(features_img2)

        d1 = self.BRIEF(self, gray1, kp1, mode='uniform', patch_size=8, n=512)  # 特征描述子
        d2 = self.BRIEF(self, gray2, kp2, mode='uniform', patch_size=8, n=512)
        matches = self.match(self, d1, d2, cross_check=True, distance_ratio=0.8)  # 匹配结果，两个id，表示该id对应特征匹配
        print('number of matches: ', matches.shape[0])

        # fig = plt.figure(figsize=(20, 10))
        # plt.suptitle('top matches')
        # ax = fig.add_subplot(1, 1, 1)
        # plot_matches(ax, gray1, gray2, np.flip(kp1, 1), np.flip(kp2, 1), matches[:100])

        # fig = plt.figure(figsize=(20, 10))
        # plt.suptitle('last')
        # ax = fig.add_subplot(1, 1, 1)
        # plot_matches(ax, gray1, gray2, np.flip(kp1, 1), np.flip(kp2, 1), matches[-3:-1])
        # plt.show()
        src_pts = np.asarray([kp1[m[0]] for m in matches], dtype=float)
        dst_pts = np.asarray([kp2[m[1]] for m in matches], dtype=float)
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 50.0)  # 要求匹配点对符合单应性变换约束
        # 挑选内点
        points1 = src_pts[mask.ravel() == 1]
        points2 = dst_pts[mask.ravel() == 1]

        kp1 = [cv2.KeyPoint(x=p[0], y=p[1], size=10) for p in points1]
        kp2 = [cv2.KeyPoint(x=p[0], y=p[1], size=10) for p in points2]
        matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(points1))]
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
        plt.figure()
        plt.suptitle('cus')
        plt.imshow(img3)
        return points1, points2


# 用Opencv库的实现，速度更快，精度更高
def standard(img1, img2):
    bak1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    bak2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # 创建orb对象
    # orb = cv2.ORB_create(500)
    orb = cv2.ORB.create(500)
    # 对ORB进行检测
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # 显示特征检测结果
    # cv2.drawKeypoints(img1, kp1, bak1, (0, 0, 255))
    # cv2.drawKeypoints(img2, kp2, bak2, (0, 0, 255))
    # plt.figure()
    # plt.suptitle('View left')
    # plt.subplot(1, 2, 1)
    # plt.imshow(img1)
    # plt.subplot(1, 2, 2)
    # plt.imshow(bak1)
    #
    # plt.figure()
    # plt.suptitle('view right')
    # plt.subplot(1, 2, 1)
    # plt.imshow(img2)
    # plt.subplot(1, 2, 2)
    # plt.imshow(bak2)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
    # plt.figure()
    # plt.suptitle('match result')
    # plt.imshow(img3)
    # plt.show()
    src_pts = np.asarray([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.asarray([kp2[m.trainIdx].pt for m in matches])
    # _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 50.0)  # 要求匹配点对符合单应性变换约束
    # 挑选内点
    points1, points2 = src_pts, dst_pts
    # points1 = src_pts[mask.ravel() == 1]
    # points2 = dst_pts[mask.ravel() == 1]
    # 用相机内参将像素坐标变为相机归一化坐标（三维坐标，在Z=1这个平面上）
    # points1n = np.dot(np.linalg.inv(K), points1)
    # points2n = np.dot(np.linalg.inv(K), points2)
    # 查看筛选后的points匹配情况
    kp1 = [cv2.KeyPoint(x=p[0], y=p[1], size=10) for p in points1]
    kp2 = [cv2.KeyPoint(x=p[0], y=p[1], size=10) for p in points2]
    matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(points1))]
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
    plt.figure()
    plt.suptitle('std')
    plt.imshow(img3)

    return points1, points2

# a, b = ORB().Run(cv2.imread('../images/1.jpg'), cv2.imread('../images/1.jpg'))
# c, d = standard(cv2.imread('../images/1.jpg'), cv2.imread('../images/1.jpg'))
# plt.show()
# standard(cv2.imread('images/chess3.jpg'), cv2.imread('images/chess.jpg'))
