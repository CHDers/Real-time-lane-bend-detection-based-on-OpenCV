import numpy as np
import pandas as pd
import cv2
import os
import glob
import matplotlib.pyplot as plt
import pickle
from moviepy.editor import VideoFileClip
from rich import print


def undistort_img():
    """
    执行相机标定并对图像去畸变
    """
    # Prepare object points 0,0,0 ... 8,5,0
    obj_pts = np.zeros((6*9, 3), np.float32)
    obj_pts[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Stores all object points & img points from all images
    objpoints = []
    imgpoints = []

    # Get directory for all calibration images
    images = glob.glob('camera_cal/*.jpg')

    for indx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)  # 检测棋盘角点
        if ret == True:
            objpoints.append(obj_pts)
            imgpoints.append(corners)
    # Test undistortion on img
    img_size = (img.shape[1], img.shape[0])

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None)  # mtx:相机的内参矩阵; dist:畸变系数; rvecs,tvecs:每张图像的旋转向量和平移向量（可选）

    dst = cv2.undistort(img, mtx, dist, None, mtx)
    # Save camera calibration for later use
    dist_pickle = {}
    dist_pickle['mtx'] = mtx
    dist_pickle['dist'] = dist
    pickle.dump(dist_pickle, open('camera_cal/cal_pickle.p', 'wb'))


def undistort(img, cal_dir='camera_cal/cal_pickle.p'):
    """对单张图像进行去畸变处理

    Args:
        img (_type_): _description_
        cal_dir (str, optional): _description_. Defaults to 'camera_cal/cal_pickle.p'.

    Returns:
        _type_: _description_
    """
    # cv2.imwrite('camera_cal/test_cal.jpg', dst)
    with open(cal_dir, mode='rb') as f:
        file = pickle.load(f)
    mtx = file['mtx']
    dist = file['dist']
    dst = cv2.undistort(img, mtx, dist, None, mtx)  # 调用OpenCV的undistort函数对图像进行去畸变处理

    return dst


def pipeline(img, s_thresh=(100, 255), sx_thresh=(15, 255)):
    """从输入图像中提取车道线相关的边缘和颜色特征, 输出一个二值图(combined_binary), 用于后续车道线检测处理

    Args:
        img (_type_): _description_
        s_thresh (tuple, optional): _description_. Defaults to (100, 255).
        sx_thresh (tuple, optional): _description_. Defaults to (15, 255).

    Returns:
        _type_: _description_
    """
    img = undistort(img)  # 使用前面定义的undistort函数对图像进行畸变校正
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)  # 将RGB图像转换为HLS色彩空间（Hue色调，Lightness亮度，Saturation饱和度）
    l_channel = hls[:, :, 1]  # 亮度通道
    s_channel = hls[:, :, 2]  # 饱和度通道
    h_channel = hls[:, :, 0]  # 色调通道(虽然此处提取了但未使用)
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1)  # Take the derivative in x  # 对亮度通道 l_channel 在 x 和 y 方向上应用 Sobel 边缘检测，目的是检测图像中边缘结构(如车道线)
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx = np.absolute(sobelx)  # 将Sobel结果取绝对值，得到无方向的梯度强度
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))  # 然后进行归一化处理（映射到0~255范围），再转换成 8 位图像格式，便于后续处理

    # Threshold x gradient   对归一化梯度图像做阈值处理，提取满足边缘强度条件的像素点，形成二值图 sxbinary
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel  对 S 通道做颜色阈值处理，提取高饱和度区域（通常是车道线的颜色特征）
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    color_binary = np.dstack(
        (np.zeros_like(sxbinary), sxbinary, s_binary)) * 255  # 将两个二值图合成为彩色图

    combined_binary = np.zeros_like(sxbinary)  # 将颜色阈值结果和 Sobel 边缘检测结果融合
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary


def perspective_warp(img,
                     dst_size=(1280, 720),
                     src=np.float32(
                         [(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)]),
                     dst=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])):
    """对输入图像进行透视变换，将原图中的车道区域转换为“鸟瞰图”视角(Bird's Eye View)，便于后续的车道线提取与拟合

    Args:
        img (_type_): _description_
        dst_size (tuple, optional): _description_. Defaults to (1280, 720).
        src (_type_, optional): _description_. Defaults to np.float32( [(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)]).
        dst (_type_, optional): _description_. Defaults to np.float32([(0, 0), (1, 0), (0, 1), (1, 1)]).

    Returns:
        _type_: _description_
    """
    img_size = np.float32([(img.shape[1], img.shape[0])])
    src = src * img_size
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)  # 计算透视变换矩阵 M，这是从 src 到 dst 的投影映射关系
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)  # 对图像进行透视变换，得到“俯视视角”图像
    return warped


def inv_perspective_warp(img,
                         dst_size=(1280, 720),
                         src=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)]),
                         dst=np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)])):
    """将鸟瞰图还原为原始视角

    Args:
        img (_type_): _description_
        dst_size (tuple, optional): _description_. Defaults to (1280, 720).
        src (_type_, optional): _description_. Defaults to np.float32([(0, 0), (1, 0), (0, 1), (1, 1)]).
        dst (_type_, optional): _description_. Defaults to np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)]).

    Returns:
        _type_: _description_
    """
    img_size = np.float32([(img.shape[1], img.shape[0])])
    src = src * img_size
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped


def get_hist(img):
    """对图像下半部分按列求和，得到水平直方图

    Args:
        img (_type_): _description_

    Returns:
        _type_: _description_
    """
    hist = np.sum(img[img.shape[0]//2:, :], axis=0)
    return hist


left_a, left_b, left_c = [], [], []
right_a, right_b, right_c = [], [], []


def sliding_window(img, nwindows=9, margin=150, minpix=1, draw_windows=True):
    """通过滑动窗口算法，从二值图像中检测并拟合出左右车道线，使用的是二次多项式拟合。输出包括绘制的图像、左右车道线的 x 坐标数组、拟合的多项式参数和用于绘图的 y 坐标数组

    Args:
        img (_type_): _description_
        nwindows (int, optional): _description_. Defaults to 9.
        margin (int, optional): _description_. Defaults to 150.
        minpix (int, optional): _description_. Defaults to 1.
        draw_windows (bool, optional): _description_. Defaults to True.
    """
    global left_a, left_b, left_c, right_a, right_b, right_c
    left_fit_ = np.empty(3)
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img))*255

    histogram = get_hist(img)  # 获取图像下半部分的水平直方图
    # find peaks of left and right halves  找到直方图左、右半部分的峰值，作为左右车道初始的 x 位置
    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        if draw_windows == True:
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (100, 255, 255), 3)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          (100, 255, 255), 3)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))


#        if len(good_right_inds) > minpix:
#            rightx_current = np.int(np.mean([leftx_current +900, np.mean(nonzerox[good_right_inds])]))
#        elif len(good_left_inds) > minpix:
#            rightx_current = np.int(np.mean([np.mean(nonzerox[good_left_inds]) +900, rightx_current]))
#        if len(good_left_inds) > minpix:
#            leftx_current = np.int(np.mean([rightx_current -900, np.mean(nonzerox[good_left_inds])]))
#        elif len(good_right_inds) > minpix:
#            leftx_current = np.int(np.mean([np.mean(nonzerox[good_right_inds]) -900, leftx_current]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    left_a.append(left_fit[0])
    left_b.append(left_fit[1])
    left_c.append(left_fit[2])

    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])

    left_fit_[0] = np.mean(left_a[-10:])
    left_fit_[1] = np.mean(left_b[-10:])
    left_fit_[2] = np.mean(left_c[-10:])

    right_fit_[0] = np.mean(right_a[-10:])
    right_fit_[1] = np.mean(right_b[-10:])
    right_fit_[2] = np.mean(right_c[-10:])

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_fitx = left_fit_[0]*ploty**2 + left_fit_[1]*ploty + left_fit_[2]
    right_fitx = right_fit_[0]*ploty**2 + right_fit_[1]*ploty + right_fit_[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
    out_img[nonzeroy[right_lane_inds],
            nonzerox[right_lane_inds]] = [0, 100, 255]

    return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty


def get_curve(img, leftx, rightx):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    y_eval = np.max(ploty)
    ym_per_pix = 30.5/720  # meters per pixel in y dimension
    xm_per_pix = 3.7/720  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature  使用曲率公式计算左右车道线在 y_eval 位置的曲率半径，单位为米
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix +
                     left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix +
                      right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    car_pos = img.shape[1]/2  # 获取图像中车辆的位置（图像中心）
    l_fit_x_int = left_fit_cr[0]*img.shape[0]**2 + \
        left_fit_cr[1]*img.shape[0] + left_fit_cr[2]  # 分别计算左右车道线在图像底部的 x 坐标（即车道线在底部的交点位置）
    r_fit_x_int = right_fit_cr[0]*img.shape[0]**2 + \
        right_fit_cr[1]*img.shape[0] + right_fit_cr[2]
    lane_center_position = (r_fit_x_int + l_fit_x_int) / 2  # 得到车道中心位置
    center = (car_pos - lane_center_position) * xm_per_pix / 10  # 计算车辆偏离车道中心的偏移量，单位为米，除以 10 是为了调整到合理范围（取决于单位设定）
    # Now our radius of curvature is in meters
    return (left_curverad, right_curverad, center)


def draw_lanes(img, left_fit, right_fit):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    color_img = np.zeros_like(img)

    left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    points = np.hstack((left, right))

    cv2.fillPoly(color_img, np.int_(points), (0, 200, 255))
    inv_perspective = inv_perspective_warp(color_img)  # 对填充图像进行逆透视变换，使其贴合原始视角
    inv_perspective = cv2.addWeighted(img, 1, inv_perspective, 0.7, 0)  # 将车道区域叠加到原图上，形成可视化效果
    return inv_perspective


def vid_pipeline(img):
    """主处理函数，接收原始图像并返回带有车道可视化和曲率信息的图像

    Args:
        img (_type_): _description_

    Returns:
        _type_: _description_
    """
    global running_avg
    global index
    img_ = pipeline(img)  # 对图像进行二值化或边缘检测处理(pipeline)是你预先定义的预处理函数）
    img_ = perspective_warp(img_)  # 将图像转换为鸟瞰图，用于车道检测
    out_img, curves, lanes, ploty = sliding_window(img_, draw_windows=False)  # 使用滑动窗口方法识别车道线并拟合曲线，返回拟合结果和中间图像
    curverad = get_curve(img, curves[0], curves[1])  # 计算车道曲率和车辆偏移量
    lane_curve = np.mean([curverad[0], curverad[1]])  # 取左右车道曲率平均值作为整体车道曲率
    img = draw_lanes(img, curves[0], curves[1])  # 将拟合好的车道区域绘制到原图上

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontColor = (0, 0, 0)
    fontSize = 0.5
    # 在图像上绘制车道曲率和偏移量
    cv2.putText(img, 'Lane Curvature: {:.0f} m'.format(
        lane_curve), (570, 620), font, fontSize, fontColor, 2)
    cv2.putText(img, 'Vehicle offset: {:.4f} m'.format(
        curverad[2]), (570, 650), font, fontSize, fontColor, 2)
    return img


if __name__ == '__main__':
    right_curves, left_curves = [],[]
    myclip = VideoFileClip('project_video.mp4')#.subclip(40,43)
    output_vid = 'output.mp4'
    clip = myclip.fl_image(vid_pipeline)
    clip.write_videofile(output_vid, audio=False)
