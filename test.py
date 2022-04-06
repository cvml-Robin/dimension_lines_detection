import cv2
import line_detection
import symmetry
import numpy as np


# 寻找最大连通域并填充
def findTower(image):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)

    tower = np.zeros(image.shape, np.uint8)

    # 最大连通域序号
    target = np.argmax(stats[1:, -1]) + 1

    for i in range(1, num_labels):

        filled = labels == i
        if i == target:
            tower[filled] = 255
        else:
            tower[filled] = 0

    contours, _ = cv2.findContours(tower, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 填充轮廓内部
    filled = np.zeros(tower.shape, np.uint8)
    filled = cv2.drawContours(filled.copy(), contours, -1, 255, -1)
    return filled


if __name__ == '__main__':
    img_name = '05.jpg'
    img_dir = 'dataset'
    save_dir = 'results'
    img = cv2.imread(img_dir + './' + img_name)

    # 二值化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    # 连通域去噪
    filledTower = findTower(binary)
    Tower = cv2.bitwise_and(binary, filledTower)
    # 直线检测
    lines = line_detection.mlsd(img)
    # 获取对称轴坐标
    r_s, theta_s = symmetry.detecting_mirrorLine(Tower)
    y_start, y_end = 0, Tower.shape[0]
    x_start = (r_s - y_start * np.sin(theta_s)) / np.cos(theta_s)
    x_end = (r_s - y_end * np.sin(theta_s)) / np.cos(theta_s)
    symmetry_axis = (x_end + x_start) / 2
    asymmetry_lines = np.zeros_like(Tower, np.uint8)
    # 去除纵向标注
    for line in lines:
        x0, y0, x1, y1 = [val for val in line]
        x_m = int((x0 + x1) / 2)
        box_w = int(symmetry_axis * 2 - (x0 + x1) / 2)
        print(x_m, box_w)
        # 判断直线对称镜像是否在图像内部
        if 0 <= box_w + 30 <= Tower.shape[1]:
            mask = np.zeros_like(Tower, np.uint8)
            mask = cv2.line(mask, (box_w, 0), (box_w, Tower.shape[0]), 255, 30)
            is_symmetry = cv2.bitwise_and(Tower, mask)
            # 获取镜像直线附近的非0像素个数
            cnt_array = (is_symmetry != 0)
            cont = cnt_array.sum()
            # 判断非0像素个数是否小于阈值，如小于，则说明该直线不对称
            if cont < 1:
                asymmetry_lines = cv2.line(asymmetry_lines, (x_m, 0), (x_m, Tower.shape[0]), 255, 10)
        # 当直线对称镜像不在图像内部时，直接去除该直线
        else:
            asymmetry_lines = cv2.line(asymmetry_lines, (x_m, 0), (x_m, Tower.shape[0]), 255, 10)

    out = cv2.bitwise_and(Tower, ~asymmetry_lines)
    temp = np.zeros_like(img, np.uint8)
    temp[:, :, 0] = out
    temp[:, :, 1] = out
    temp[:, :, 2] = out
    temp = ~cv2.bitwise_and(img, temp)
    print(Tower.shape[0])
    # 保存图片
    cv2.namedWindow('out', cv2.WINDOW_NORMAL)
    cv2.imshow('out', temp)
    cv2.waitKey()
    cv2.destroyAllWindows()
