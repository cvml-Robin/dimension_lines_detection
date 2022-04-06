import numpy as np
import tensorflow as tf
import cv2
import random


def mlsd(image, args):
    interpreter, input_details, output_details = load_tflite(args.tflite_path)
    segments = pred_squares(image, interpreter, input_details,
                            output_details,
                            [args.input_size, args.input_size],
                            args.score_thr)
    draw_lines = draw_output(image, segments)
    return segments, draw_lines


def load_tflite(tflite_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details


def pred_squares(image, interpreter, input_details, output_details, input_shape, score):
    h, w, _ = image.shape
    original_shape = [h, w]

    resized_image = np.concatenate([cv2.resize(image, (input_shape[0], input_shape[1]), interpolation=cv2.INTER_AREA),
                                    np.ones([input_shape[0], input_shape[1], 1])], axis=-1)
    batch_image = np.expand_dims(resized_image, axis=0).astype('float32')
    interpreter.set_tensor(input_details[0]['index'], batch_image)
    interpreter.invoke()

    pts = interpreter.get_tensor(output_details[0]['index'])[0]
    pts_score = interpreter.get_tensor(output_details[1]['index'])[0]
    vmap = interpreter.get_tensor(output_details[2]['index'])[0]

    start = vmap[:, :, :2]  # (x, y)
    end = vmap[:, :, 2:]  # (x, y)
    dist_map = np.sqrt(np.sum((start - end) ** 2, axis=-1))

    junc_list = []
    segments_list = []
    for junc, _score in zip(pts, pts_score):
        y, x = junc
        distance = dist_map[y, x]
        if _score > score and distance > 20.0:
            junc_list.append([x, y])
            disp_x_start, disp_y_start, disp_x_end, disp_y_end = vmap[y, x, :]
            d_arrow = 1.0
            x_start = x + d_arrow * disp_x_start
            y_start = y + d_arrow * disp_y_start
            x_end = x + d_arrow * disp_x_end
            y_end = y + d_arrow * disp_y_end
            segments_list.append([x_start, y_start, x_end, y_end])

    segments = np.array(segments_list)

    # get unique lines
    point = np.array([[0, 0]])
    point = point[0]
    start = segments[:, :2]
    end = segments[:, 2:]
    diff = start - end
    a = diff[:, 1]
    b = -diff[:, 0]
    c = a * start[:, 0] + b * start[:, 1]

    d = np.abs(a * point[0] + b * point[1] - c) / np.sqrt(a ** 2 + b ** 2 + 1e-10)
    theta = np.arctan2(diff[:, 0], diff[:, 1]) * 180 / np.pi
    theta[theta < 0.0] += 180
    hough = np.concatenate([d[:, None], theta[:, None]], axis=-1)

    d_quant = 1
    theta_quant = 2
    hough[:, 0] //= d_quant
    hough[:, 1] //= theta_quant
    _, indices, counts = np.unique(hough, axis=0, return_index=True, return_counts=True)

    acc_map = np.zeros([512 // d_quant + 1, 360 // theta_quant + 1], dtype='float32')
    idx_map = np.zeros([512 // d_quant + 1, 360 // theta_quant + 1], dtype='int32') - 1
    yx_indices = hough[indices, :].astype('int32')
    acc_map[yx_indices[:, 0], yx_indices[:, 1]] = counts
    idx_map[yx_indices[:, 0], yx_indices[:, 1]] = indices

    acc_map_np = acc_map
    acc_map = acc_map[None, :, :, None]

    # fast suppression using tensorflow op
    acc_map = tf.constant(acc_map, dtype=tf.float32)
    max_acc_map = tf.keras.layers.MaxPool2D(pool_size=(5, 5), strides=1, padding='same')(acc_map)
    acc_map = acc_map * tf.cast(tf.math.equal(acc_map, max_acc_map), tf.float32)
    flatten_acc_map = tf.reshape(acc_map, [1, -1])
    topk_values, topk_indices = tf.math.top_k(flatten_acc_map, k=len(pts))
    _, h, w, _ = acc_map.shape
    y = tf.expand_dims(topk_indices // w, axis=-1)
    x = tf.expand_dims(topk_indices % w, axis=-1)
    yx = tf.concat([y, x], axis=-1)
    yx = yx[0].numpy()
    indices = idx_map[yx[:, 0], yx[:, 1]]
    topk_values = topk_values.numpy()[0]
    basis = 5 // 2

    merged_segments = []
    for yx_pt, max_indice, value in zip(yx, indices, topk_values):
        y, x = yx_pt
        if max_indice == -1 or value == 0:
            continue
        segment_list = []
        for y_offset in range(-basis, basis + 1):
            for x_offset in range(-basis, basis + 1):
                indice = idx_map[y + y_offset, x + x_offset]
                cnt = int(acc_map_np[y + y_offset, x + x_offset])
                if indice != -1:
                    segment_list.append(segments[indice])
                if cnt > 1:
                    check_cnt = 1
                    current_hough = hough[indice]
                    for new_indice, new_hough in enumerate(hough):
                        if (current_hough == new_hough).all() and indice != new_indice:
                            segment_list.append(segments[new_indice])
                            check_cnt += 1
                        if check_cnt == cnt:
                            break
        group_segments = np.array(segment_list).reshape([-1, 2])
        sorted_group_segments = np.sort(group_segments, axis=0)
        x_min, y_min = sorted_group_segments[0, :]
        x_max, y_max = sorted_group_segments[-1, :]

        deg = theta[max_indice]
        # 保留竖直直线
        if 170 <= deg <= 180:
            merged_segments.append([x_min, y_max, x_max, y_min])
        elif 0 <= deg <= 10:
            merged_segments.append([x_min, y_min, x_max, y_max])

    # 2. get intersections
    new_segments = np.array(merged_segments)  # (x1, y1, x2, y2)

    new_segments[:, 0] = new_segments[:, 0] * 2 / input_shape[1] * original_shape[1]
    new_segments[:, 1] = new_segments[:, 1] * 2 / input_shape[0] * original_shape[0]
    new_segments[:, 2] = new_segments[:, 2] * 2 / input_shape[1] * original_shape[1]
    new_segments[:, 3] = new_segments[:, 3] * 2 / input_shape[0] * original_shape[0]

    return new_segments


def draw_output(image, segments):
    line_image = image.copy()
    line_thick = 5
    # output > line array
    for line in segments:
        x_start, y_start, x_end, y_end = [int(val) for val in line]
        cv2.line(line_image, (x_start, y_start), (x_end, y_end),
                 [random.randrange(255), random.randrange(255), random.randrange(255)], line_thick)

    return line_image
