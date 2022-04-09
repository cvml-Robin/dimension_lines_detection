import cv2
import numpy as np
import random


def combine(_lines, _p0, _p1, _vertical):
    indz = 0
    for k in _p1:
        mark, mindis = minDistance(_lines[_p0], _lines[k], _vertical)
        if mindis <= 15 or mark:
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


def draw_output(image, segments):
    line_image = image.copy()
    line_thick = 5
    # output > line array
    for line in segments:
        x_start, y_start, x_end, y_end, _ = [int(val) for val in line]
        cv2.line(line_image, (x_start, y_start), (x_end, y_end),
                 [random.randrange(255), random.randrange(255), random.randrange(255)], line_thick)

    return line_image


def detectlines(_gray, _img):
    # 创建一个LSD对象
    lsd = cv2.createLineSegmentDetector(scale=1)
    # 执行检测结果
    dlines = lsd.detect(_gray)
    lines = dlines[0].reshape(-1, 4)
    vertical = []
    horizontal = []
    for a in lines:
        x0, y0, x1, y1 = a
        x_mid = (x0 + x1) / 2
        y_mid = (y0 + y1) / 2
        theta = np.arctan((y1 - y0) / (x1 - x0 + np.spacing(1))) * 180 / np.pi
        if np.abs(theta) <= 2:
            horizontal.append([x0, y0, x1, y1, y_mid])
        elif np.abs(theta) >= 88:
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
    line_thick = 5
    draw_lines_ver = _img.copy()
    draw_lines_hor = _img.copy()
    out_ver = []
    out_hor = []
    for line in vertical:
        x_start, y_start, x_end, y_end, _ = [int(val) for val in line]
        if np.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2) >= 200:
            out_ver.append([x_start, y_start, x_end, y_end])
            cv2.line(draw_lines_ver, (x_start, y_start), (x_end, y_end),
                     [random.randrange(255), random.randrange(255), random.randrange(255)], line_thick)

    for line in horizontal:
        x_start, y_start, x_end, y_end, _ = [int(val) for val in line]
        if np.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2) >= 200:
            out_hor.append([x_start, y_start, x_end, y_end])
            cv2.line(draw_lines_hor, (x_start, y_start), (x_end, y_end),
                     [random.randrange(255), random.randrange(255), random.randrange(255)], line_thick)
    draw_lines_ver = draw_output(_img, vertical)
    draw_lines_hor = draw_output(_img, horizontal)
    return out_ver, out_hor, draw_lines_ver, draw_lines_hor
