import argparse
import os
import time
import cv2
import numpy as np
from utils import lsd_lines, symmetry

# config
parser = argparse.ArgumentParser()
# 数据集与输出路径
parser.add_argument('--input_dir', default='demo', type=str)
parser.add_argument('--results_dir', default='results', type=str)


def gaussian(u, sigma, n):
    x = np.arange(n)
    y = np.exp(-(x - u) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
    return y


if __name__ == '__main__':
    start_time = time.time()
    args = parser.parse_args()
    data_dir = args.input_dir
    save_dir = args.results_dir
    img_num = 0
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for img_dir in os.listdir(r'./' + data_dir):
        if not os.path.exists(save_dir + '/' + img_dir):
            os.makedirs(save_dir + '/' + img_dir)
        for img_name in os.listdir(r'./' + data_dir + '/' + img_dir):
            img_num += 1
            gray = cv2.imread((data_dir + '/' + img_dir + '/' + img_name), 0)
            img = np.zeros((gray.shape[0], gray.shape[1], 3), np.uint8)
            img[:, :, 0] = gray
            img[:, :, 1] = gray
            img[:, :, 2] = gray
            # 二值化
            _, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
            # 直线检测
            lines, _, draw_lines, _ = lsd_lines.detectlines(gray, img)
            # 获取对称轴坐标
            symmetry_axis, r, theta, draw_symmetry = symmetry.detecting_mirrorLine(img, gray)
            temp = binary.copy()
            # 去除纵向标注
            for line in lines:
                x0, y0, x1, y1 = [int(val) for val in line]
                x0_mir = int(symmetry_axis * 2) - x0
                x1_mir = int(symmetry_axis * 2) - x1
                x_mid = (x0 + x1) >> 1
                x_mir_mid = (x0_mir + x1_mir) >> 1
                # 判断直线对称镜像是否在图像内部
                is_inside = (np.abs(x_mir_mid - binary.shape[1] >> 1) +
                             np.abs((x1 - x0) >> 1)) < binary.shape[1] >> 1
                if is_inside:
                    mask = gaussian(x_mid, 3, binary.shape[1]).reshape(1, -1)
                    mask_mir = gaussian(x_mir_mid, 3, binary.shape[1]).reshape(1, -1)
                    # 获取直线与其镜像直线附近的非0像素个数
                    cnt = np.multiply(binary, mask).sum()
                    cnt_mir = np.multiply(binary, mask_mir).sum()
                    # 比较直线与其镜像直线附近点数
                    if cnt_mir > (cnt * 0.2):
                        continue
                temp = cv2.line(temp, (x0, y0), (x1, y1), 0, 15)
            out = np.zeros_like(img, np.uint8)
            out[:, :, 0] = temp
            out[:, :, 1] = temp
            out[:, :, 2] = temp
            out = ~cv2.bitwise_and(~img, out)
            rows, cols = img.shape[0], img.shape[1]
            # 拼接原图、对称轴图、直线检测图与最终结果图
            result = np.zeros((rows * 2, cols * 2, 3), np.uint8)
            result[0:rows, 0:cols, :] = img
            result[rows:rows * 2, 0:cols, :] = draw_lines
            result[0:rows, cols:cols * 2, :] = draw_symmetry
            result[rows:rows * 2, cols:cols * 2, :] = out
            # 保存图片
            save_path = save_dir + '/' + img_dir + '/output_' + img_name
            print('saving ' + img_dir + '_' + img_name)
            cv2.imwrite(save_path, result)
    end_time = time.time()
    total_time = end_time - start_time
    aver_time = total_time / img_num
    print('程序运行总计耗时%.2f秒！' % total_time)
    print('处理每张图片平均耗时%.2f秒！' % aver_time)
