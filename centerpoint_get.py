import cv2
import math
import numpy as np

def azimuthAngle( x1,  y1,  x2,  y2):
    angle = 0.0
    dx = x2 - x1
    dy = y2 - y1
    if x2 == x1:
        angle = math.pi / 2.0
        if y2 == y1:
            angle = 0.0
        elif y2 < y1:
            angle = 3.0 * math.pi / 2.0
    elif x2 > x1 and y2 > y1:
        angle = math.atan(dx / dy)
    elif x2 > x1 and y2 < y1:
        angle = math.pi / 2 + math.atan(-dy / dx)
    elif x2 < x1 and y2 < y1:
        angle = math.pi + math.atan(dx / dy)
    elif x2 < x1 and y2 > y1:
        angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
    return (angle * 180 / math.pi)

#
# def azimuthAngle_mask(mask_contours , mask_contours):
#     """
#     xs: 所有待计算像素点
#     mask:mask像素点
#     """
#     angle =
#     xs = points[0]
#     ys = points[1]
#     mask_x = xs[:, np.newaxis] - mask_[:, 0]
#     mask_y = ys[:, np.newaxis] - mask_[:, 1]


def center_contour_angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)
    # print(angle2)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle

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
    x = x / area
    y = y / area
    return [int(x), int(y)]


def get_single_centerpoint(mask):
        contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour.sort(key=lambda x: cv2.contourArea(x), reverse=True) #only save the biggest one
        '''debug IndexError: list index out of range'''
        count = contour[0][:, 0, :]
        try:
            center = get_centerpoint(count)
        except:
            x,y = count.mean(axis=0)
            center=[int(x), int(y)]
        return center, contour

def contour_center_rays_get(center, contour):
    contour_center_rays = np.zeros(36)
    contour_angle = np.zeros(contour.shape[0])
    contour_distance = np.zeros(contour.shape[0])
    p1 = np.array([center[0], center[1]])
    # p2 = np.array([center[0], center[1]+5])
    for index, cont in enumerate(contour):
        p3 = np.array(cont)
        # p1_p2 = np.append(p1, p2)
        # p1_p3 = np.append(p1, p3)
        # angle_i = center_contour_angle(p1_p2, p1_p3)
        angle_i = azimuthAngle(p1[0], p1[1], p3[0], p3[1])
        contour_angle[index] = angle_i
        contour_distance[index] = np.linalg.norm(p1 - p3)
    for ang in range(36):
        contour_angle_ang = np.abs(contour_angle - ang*10)
        contour_angle_ang_min = np.min(contour_angle_ang)
        if contour_angle_ang_min < 5:
            contour_angle_ang_min_index = np.where(contour_angle_ang == contour_angle_ang_min)
            contour_distance_index = contour_distance[contour_angle_ang_min_index]
            contour_distance_min_index = np.min(contour_distance_index)
            contour_center_rays[ang] = contour_distance_min_index
    return contour_center_rays

def get_36_coordinates(location_to_mask, pos_mask_contour):
    # ct = pos_mask_contour[:, 0, :]
    ct = np.squeeze(pos_mask_contour, 1)
    x = ct[:, 0, np.newaxis] - location_to_mask[:,0]
    y = ct[:, 1, np.newaxis] - location_to_mask[:,1]
    angle = np.arctan2(x, y)*180/np.pi
    # angle = torch.atan2(x, y) * 180 / np.pi
    angle[angle < 0] += 360
    # angle = angle.int()
    angle = np.array(angle,dtype=np.int)
    dist = np.sqrt(x ** 2 + y ** 2)
    # dist = torch.sqrt(x ** 2 + y ** 2)
    # angle, idx = torch.sort(angle)
    # idx = np.argsort(angle, 0)
    # angle = np.sort(angle, 0)
    # dist = dist[idx]
    distances = []
    for point_i in range(len(location_to_mask)):
        contour_center_rays = np.zeros(36)
        contour_angle = angle[:, point_i]
        contour_distance = dist[:, point_i]
        for ang in range(36):
            contour_angle_ang = np.abs(contour_angle - ang*10)
            contour_angle_ang_min = np.min(contour_angle_ang)
            if contour_angle_ang_min < 5:
                contour_angle_ang_min_index = np.where(contour_angle_ang == contour_angle_ang_min)
                contour_distance_index = contour_distance[contour_angle_ang_min_index]
                contour_distance_min_index = np.min(contour_distance_index)
                contour_center_rays[ang] = contour_distance_min_index
        distances.append(contour_center_rays)

    return distances

def get_mask_sample_region(gt_bb, mask_center, strides, num_points_per, gt_xs, gt_ys, radius=1):
    center_y = mask_center[..., 1]
    center_x = mask_center[..., 0]
    # center_gt = gt_bb.new_zeros(gt_bb.shape)
    center_gt = np.zeros(gt_bb.shape)
    #no gt
    if center_x[..., 0].sum() == 0:
        return gt_xs.new_zeros(gt_xs.shape, dtype=np.uint8)

    beg = 0
    for level,n_p in enumerate(num_points_per):
        end = beg + n_p
        stride = strides[level][0] * radius
        xmin = center_x[beg:end] - stride
        ymin = center_y[beg:end] - stride
        xmax = center_x[beg:end] + stride
        ymax = center_y[beg:end] + stride
        # limit sample region in gt
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
    inside_gt_bbox_mask = np.min(center_bbox, axis=-1) > 0 # 上下左右都>0 就是在bbox里面
    return inside_gt_bbox_mask