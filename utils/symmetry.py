import cv2
import matplotlib.pyplot as plt
import numpy as np

# create SIFT object ((a feature detection algorithm))
# noinspection PyUnresolvedReferences
sift = cv2.SIFT_create()
# create BFMatcher object
bf = cv2.BFMatcher()


def detecting_mirrorLine(img, binary):
    # create mirror object
    mirror = MirrorSymmetryDetection(binary)

    # extracting and Matching a pair of symmetric features
    matchpoints = mirror.find_matchpoints()

    # get r, tehta (polar coordinates) of the midpoints of all pair of symmetric features
    points_r, points_theta = mirror.find_points_r_theta(matchpoints)

    # find the best one with the highest vote
    image_hexbin = plt.hexbin(points_r, points_theta, bins=200, cmap=plt.cm.Spectral_r)
    sorted_vote = sort_hexbin_by_votes(image_hexbin)
    r, theta = find_coordinate_maxhexbin(sorted_vote)
    y_start, y_end = 0, binary.shape[0]
    x_start = (r - y_start * np.sin(theta)) / np.cos(theta)
    x_end = (r - y_end * np.sin(theta)) / np.cos(theta)
    symmetry_axis = (x_end + x_start) / 2
    draw_lines = draw_mirrorLine(img, r, theta)
    return symmetry_axis, r, theta, draw_lines


def find_coordinate_maxhexbin(sorted_vote):
    # Try to find the x and y coordinates of the hexbin with max count
    for k, _ in sorted_vote.items():
        return k[0], k[1]


def sort_hexbin_by_votes(image_hexbin):
    # Sort hexbins by decreasing count. (lower vote)
    counts = image_hexbin.get_array()
    verts = image_hexbin.get_offsets()  # coordinates of each hexbin
    output = {}

    for offc in range(verts.shape[0]):
        binx, biny = verts[offc][0], verts[offc][1]
        if counts[offc]:
            output[(binx, biny)] = counts[offc]
    return {k: v for k, v in sorted(output.items(), key=lambda item: item[1], reverse=True)}


def angle_with_x_axis(pi, pj):  # 公式在文件里解释
    # get the difference between point p1 and p2
    x, y = pi[0] - pj[0], pi[1] - pj[1]

    if x == 0:
        return np.pi / 2

    angle = np.arctan(y / x)
    if angle < 0:
        angle += np.pi
    return angle


def draw_mirrorLine(image, r, theta):
    mirrorline = image.copy()
    y_start, y_end = 0, mirrorline.shape[0]
    x_start = int((r - y_start * np.sin(theta)) / np.cos(theta))
    x_end = int((r - y_end * np.sin(theta)) / np.cos(theta))
    cv2.line(mirrorline, (x_start, y_start), (x_end, y_end),
             (255, 0, 0), 5)
    return mirrorline


class MirrorSymmetryDetection:
    def __init__(self, image):
        # b, g, r = cv2.split(image)  # get b,g,r
        # self.image = cv2.merge([r, g, b])  # switch it to rgb
        self.image = image
        self.reflected_image = np.fliplr(self.image)  # Flipped version of image

        # find the keypoints and descriptors with SIFT
        self.kp1, self.des1 = sift.detectAndCompute(self.image, None)
        self.kp2, self.des2 = sift.detectAndCompute(self.reflected_image, None)

    def find_matchpoints(self):
        # use BFMatcher.knnMatch() to get （k=2）matches
        matches = bf.knnMatch(self.des1, self.des2, k=2)
        # these matches are equivalent only one need be recorded
        matchpoints = [item[0] for item in matches]

        # sort to determine the dominant symmetries
        matchpoints = sorted(matchpoints, key=lambda x: x.distance)

        return matchpoints

    def find_points_r_theta(self, matchpoints: list):
        # Get r, tehta of the midpoints of all pair of symmetric features
        points_r = []  # list of r for each point
        points_theta = []  # list of theta for each point
        for match in matchpoints:

            point = self.kp1[match.queryIdx]  # queryIdx is an index into one set of keypoints, (origin image)
            mirpoint = self.kp2[match.trainIdx]  # trainIdx is an index into the other set of keypoints (fliped image)

            mirpoint.angle = np.deg2rad(mirpoint.angle)  # Normalise orientation
            mirpoint.angle = np.pi - mirpoint.angle
            # convert angles to positive
            if mirpoint.angle < 0.0:
                mirpoint.angle += 2 * np.pi

            # pt: coordinates of the keypoints x:pt[0], y:pt[1]
            # change x, not y
            mirpoint.pt = (self.reflected_image.shape[1] - mirpoint.pt[0], mirpoint.pt[1])

            # get θij: the angle this line subtends with the x-axis.
            theta = angle_with_x_axis(point.pt, mirpoint.pt)

            # midpoit (xc,yc) are the image centred co-ordinates of the mid-point of the line joining pi and pj
            xc, yc = (point.pt[0] + mirpoint.pt[0]) / 2, (point.pt[1] + mirpoint.pt[1]) / 2
            r = xc * np.cos(theta) + yc * np.sin(theta)

            points_r.append(r)
            points_theta.append(theta)

        return points_r, points_theta  # polar
