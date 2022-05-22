import numpy as np
import cv2


def get_centerpoint(lis):
    area = 0.0
    x, y = 0.0, 0.0
    a = len(lis)
    for i in range(a):
        lat = lis[i][0]
        lng = lis[i][1]
        if i == 0:
            lat1 = lis[-1][0]
            lng1 = lis[-1][1]
        else:
            lat1 = lis[i - 1][0]
            lng1 = lis[i - 1][1]
        fg = (lat * lng1 - lng * lat1) / 2.0
        area += fg
        x += fg * (lat + lat1) / 3.0
        y += fg * (lng + lng1) / 3.0
    return x, y, area


def get_mask_sample_region(gt_bb, mask_center, strides, num_points_per, gt_xs, gt_ys, radius=1):
    """

    :param gt_bb:
    :param mask_center:
    :param strides:
    :param num_points_per:
    :param gt_xs:
    :param gt_ys:
    :param radius:
    :return:
    """
    center_y = mask_center[..., 0]
    center_x = mask_center[..., 1]

    center_gt = np.zeros_like(gt_bb, dtype=np.float32)

    if center_x[..., 0].sum() == 0:
        return np.zeros_like(gt_xs, dtype=np.uint8)

    beg = 0
    for level, n_p in enumerate(num_points_per):
        end = beg + n_p
        stride = strides[level] * radius
        xmin = center_x[beg:end] - stride
        ymin = center_y[beg:end] - stride
        xmax = center_x[beg:end] + stride
        ymax = center_y[beg:end] + stride

        center_gt[beg:end, :, 0] = np.where(xmin > gt_bb[beg:end, :, 0], xmin, gt_bb[beg:end, :, 0])
        center_gt[beg:end, :, 1] = np.where(ymin > gt_bb[beg:end, :, 1], ymin, gt_bb[beg:end, :, 1])
        center_gt[beg:end, :, 2] = np.where(xmax > gt_bb[beg:end, :, 2], gt_bb[beg:end, :, 2], xmax)
        center_gt[beg:end, :, 3] = np.where(ymax > gt_bb[beg:end, :, 3], gt_bb[beg:end, :, 3], ymax)
        beg = end

    left = gt_xs[:, np.newaxis] - center_gt[..., 0]
    right = center_gt[..., 2] - gt_xs[:, np.newaxis]
    top = gt_ys[:, np.newaxis] - center_gt[..., 1]
    bottom = center_gt[..., 3] - gt_ys[:, np.newaxis]
    center_bbox = np.stack((left, top, right, bottom), -1)
    inside_gt_bbox_mask = np.min(center_bbox, axis=-1) > 0  # 上下左右都>0 就是在bbox里面
    return inside_gt_bbox_mask


def get_36_coordinates(c_x, c_y, pos_mask_contour):
    ct = pos_mask_contour[:, 0, :]
    x = ct[:, 0] - c_x
    y = ct[:, 1] - c_y
    # angle = np.arctan2(x, y)*180/np.pi
    angle = np.arctan2(x, y) * 180 / np.pi
    angle[angle < 0] += 360
    angle = angle.astype(np.int32)
    # dist = np.sqrt(x ** 2 + y ** 2)
    dist = np.sqrt(x ** 2 + y ** 2)
    idx = np.argsort(angle)
    angle = np.sort(angle)
    dist = dist[idx]
    dist[dist < 1] = 1
    # 生成36个角度
    new_coordinate = {}
    for i in range(0, 360, 10):
        if i in angle:
            d = dist[angle == i].max()
            new_coordinate[i] = d
        elif i + 1 in angle:
            d = dist[angle == i + 1].max()
            new_coordinate[i] = d
        elif i - 1 in angle:
            d = dist[angle == i - 1].max()
            new_coordinate[i] = d
        elif i + 2 in angle:
            d = dist[angle == i + 2].max()
            new_coordinate[i] = d
        elif i - 2 in angle:
            d = dist[angle == i - 2].max()
            new_coordinate[i] = d
        elif i + 3 in angle:
            d = dist[angle == i + 3].max()
            new_coordinate[i] = d
        elif i - 3 in angle:
            d = dist[angle == i - 3].max()
            new_coordinate[i] = d

    distances = np.zeros(36)
    masks = np.ones(36)

    for a in range(0, 360, 10):
        if not a in new_coordinate.keys():
            new_coordinate[a] = 1
            distances[a // 10] = 1
            masks[a//10] = 0
        else:
            distances[a // 10] = new_coordinate[a]
    # for idx in range(36):
    #     dist = new_coordinate[idx * 10]
    #     distances[idx] = dist

    return distances, masks


def get_36_coordinates_new(c_x, c_y, pos_mask_contour):
    ct = pos_mask_contour[:, 0, :]
    x = ct[:, 0] - c_x
    y = ct[:, 1] - c_y
    # angle = np.arctan2(x, y)*180/np.pi
    angle = np.arctan2(x, y) * 180 / np.pi
    angle[angle < 0] += 360
    angle = angle.astype(np.int32)
    # dist = np.sqrt(x ** 2 + y ** 2)
    dist = np.sqrt(x ** 2 + y ** 2)
    idx = np.argsort(angle)
    angle = np.sort(angle)
    dist = dist[idx]

    dist[dist < 1] = 1

    # 生成36个角度
    new_coordinate = {}
    for i in range(0, 360, 10):
        if i in angle:
            d = dist[angle == i].max()
            new_coordinate[i] = d
        elif i + 1 in angle:
            d = dist[angle == i+1].max()
            new_coordinate[i] = d
        elif i - 1 in angle:
            d = dist[angle == i-1].max()
            new_coordinate[i] = d
        elif i + 2 in angle:
            d = dist[angle == i+2].max()
            new_coordinate[i] = d
        elif i - 2 in angle:
            d = dist[angle == i-2].max()
            new_coordinate[i] = d
        elif i + 3 in angle:
            d = dist[angle == i+3].max()
            new_coordinate[i] = d
        elif i - 3 in angle:
            d = dist[angle == i-3].max()
            new_coordinate[i] = d

    distances = np.zeros(36)

    mean_distance = 0
    for a in new_coordinate.keys():
        mean_distance += new_coordinate[a]

    if len(new_coordinate) == 0:
        mean_distance = 1
    else:
        mean_distance = mean_distance / len(new_coordinate)

    for a in range(0, 360, 10):
        if not a in new_coordinate.keys():
            new_coordinate[a] = mean_distance
            distances[a//10] = mean_distance
        else:
            distances[a//10] = new_coordinate[a]
    # for idx in range(36):
    #     dist = new_coordinate[idx * 10]
    #     distances[idx] = dist

    return distances, new_coordinate


def get_single_centerpoint(mask):
    _, contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour.sort(key=lambda x: cv2.contourArea(x), reverse=True) # only save the biggest one
    '''debug IndexError: list index out of range'''
    count = contour[0][:, 0, :]

    x, y, area = get_centerpoint(count)
    if area > 0:
        center = [int(x/area), int(y/area)]
    else:
        x, y = count.mean(axis=0)
        center = [int(x), int(y)]
    return center, contour