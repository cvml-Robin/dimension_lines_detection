import cv2
import numpy as np
import random
import time


def combine(_lines, _p0, _p1, _vertical):
    indz = 0
    for k in _p1:
        mark, mindis = minDistance(_lines[_p0], _lines[k], _vertical)
        if mindis <= 8 or mark:
            return True, indz, k
        indz = indz + 1
    return False, 0, 0


def minDistance(_line_0, _line_1, _vertical=True):
    _x0_0, _y0_0, _x1_0, _y1_0, _ = _line_0
    _x0_1, _y0_1, _x1_1, _y1_1, _ = _line_1
    distance = np.zeros(4)
    distance[0] = np.sqrt((_x0_0 - _x0_1) ** 2 + (_y0_0 - _y0_1) ** 2)
    distance[1] = np.sqrt((_x1_0 - _x0_1) ** 2 + (_y1_0 - _y0_1) ** 2)
    distance[2] = np.sqrt((_x0_0 - _x1_1) ** 2 + (_y0_0 - _y1_1) ** 2)
    distance[3] = np.sqrt((_x1_0 - _x1_1) ** 2 + (_y1_0 - _y1_1) ** 2)
    if _vertical:
        min_0, max_0 = min(_y0_0, _y1_0), max(_y0_0, _y1_0)
        min_1, max_1 = min(_y0_1, _y1_1), max(_y0_1, _y1_1)
        if min_1 <= _y0_0 <= max_1 or min_1 <= _y1_0 <= max_1 or min_0 <= _y0_1 <= max_0 or min_0 <= _y1_1 <= max_0:
            return True, np.min(distance)
    else:
        min_0, max_0 = min(_x0_0, _x1_0), max(_x0_0, _x1_0)
        min_1, max_1 = min(_x0_1, _x1_1), max(_x0_1, _x1_1)
        if min_1 <= _x0_0 <= max_1 or min_1 <= _x1_0 <= max_1 or min_0 <= _x0_1 <= max_0 or min_0 <= _x1_1 <= max_0:
            return True, np.min(distance)
    return False, np.min(distance)


if __name__ == '__main__':
    start_time = time.time()
    gray = cv2.imread('01.png', 0)
    _, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    img = np.zeros((gray.shape[0], gray.shape[1], 3), np.uint8)
    img[:, :, 0] = gray
    img[:, :, 1] = gray
    img[:, :, 2] = gray
    # 创建一个LSD对象
    lsd = cv2.createLineSegmentDetector(scale=1)
    # 执行检测结果
    dlines = lsd.detect(gray)
    lines = dlines[0].reshape(-1, 4)
    vertical = []
    horizontal = []
    for a in lines:
        x0, y0, x1, y1 = a
        x_mid = (x0 + x1) / 2
        y_mid = (y0 + y1) / 2
        theta = np.arctan((y1 - y0) / (x1 - x0 + np.spacing(1))) * 180 / np.pi
        if np.abs(theta) <= 5:
            horizontal.append([x0, y0, x1, y1, y_mid])
        elif np.abs(theta) >= 85:
            vertical.append([x0, y0, x1, y1, x_mid])
    # 合并竖直线
    for i in range(0, len(vertical)):
        temp = []
        for j in range(i + 1, len(vertical)):
            line0 = vertical[i]
            line1 = vertical[j]
            if np.abs(line0[4] - line1[4]) <= 5:
                temp.append(j)
        inds = []
        while temp:
            flage, index0, index1 = combine(vertical, i, temp, True)
            if not flage:
                break
            line = np.array([vertical[i], vertical[index1]])
            y_arr = np.array([line[0, 1], line[0, 3], line[1, 1], line[1, 3]])
            start, end = np.argmin(y_arr), np.argmax(y_arr)
            vertical[i] = [line[start // 2, start % 2 * 2], line[start // 2, start % 2 * 2 + 1],
                           line[end // 2, end % 2 * 2], line[end // 2, end % 2 * 2 + 1], vertical[i][4]]
            inds.append(index1)
            temp.pop(index0)

        inds.sort(reverse=True)
        for index in inds:
            vertical.pop(index)
    # 合并水平线
    for i in range(0, len(horizontal)):
        temp = []
        for j in range(i + 1, len(horizontal)):
            line0 = horizontal[i]
            line1 = horizontal[j]
            if np.abs(line0[4] - line1[4]) <= 5:
                temp.append(j)
        inds = []
        while temp:
            flage, index0, index1 = combine(horizontal, i, temp, False)
            if not flage:
                break
            line = np.array([horizontal[i], horizontal[index1]])
            x_arr = np.array([line[0, 0], line[0, 2], line[1, 0], line[1, 2]])
            start, end = np.argmin(x_arr), np.argmax(x_arr)
            horizontal[i] = [line[start // 2, start % 2 * 2], line[start // 2, start % 2 * 2 + 1],
                             line[end // 2, end % 2 * 2], line[end // 2, end % 2 * 2 + 1], horizontal[i][4]]
            inds.append(index1)
            temp.pop(index0)

        inds.sort(reverse=True)
        for index in inds:
            horizontal.pop(index)

    # for a in horizontal:
    #     x0, y0, x1, y1, _ = [int(val) for val in a]
    #     cv2.line(img, (x0, y0), (x1, y1),
    #              (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 2)
    # cv2.namedWindow('out', cv2.WINDOW_NORMAL)
    # cv2.imshow('out', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    end_time = time.time()
    print(end_time - start_time)
