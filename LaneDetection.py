import numpy as np
import cv2
import Settings
from collections import deque


class LaneDetection:

    def __init__(self, max_array):
        self.max_array = max_array
        self.curve_array = deque(maxlen=max_array)
        self.ster_array = deque(maxlen=max_array)
        self.center_array = deque(maxlen=max_array)
        self.radius = 0



    def detect(self, img):
        imgCanny = self.thresholding(img)
        wrap = self.perspectivae_wrap(imgCanny)
        img_c = self.draw_circle(img)
        curves, abc, finalImg = self.sliding_window(wrap)

        self.curve_array.append([curves[0], curves[1]])
        new_curves = self.calculate_average_curve(self.curve_array)

        try:
            curverad = self.get_curve(finalImg, new_curves[0], new_curves[1])
            lane_center = np.mean([curverad[0], curverad[1]])
            #lane_cross= np.mean([cross[0], cross[1]])
            imgFinal = self.draw_lanes(img, new_curves[0], new_curves[1])

            #self.ster_array.append(lane_cross//200)
            self.center_array.append(lane_center //200)
            img = self.connect_img(img, imgFinal)

        except:
            pass


        return img, np.mean(self.ster_array), np.mean(self.center_array)

    def fill_gaps(self, img):
        try:
            lines = cv2.HoughLinesP(img, 1, np.pi/180, 70,np.array ([ ]), 80, 10)

            for i in lines:
                coords = i[0]
                cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255,255,255,], 5)
        except:
            pass

        return img



    def calculate_average_curve(self, curves):
        if len(curves) !=0:
            average_curve_L = np.zeros(np.array(curves[0][0]).shape)
            average_curve_R = np.zeros(np.array(curves[0][1]).shape)


            for curve in curves:
                average_curve_L = np.array(curve[0])+ average_curve_L
                average_curve_R = np.array(curve[1])+ average_curve_R

            average_curve_L = average_curve_L/len(curves)
            average_curve_R = average_curve_R/len(curves)


            return (average_curve_L, average_curve_R)



    def inv_perspective_warp(self, img):
        img_size = np.float32([(img.shape[1], img.shape[0])])

        src = Settings.DST * img_size
        dst = Settings.SRC * np.float32((Settings.MONITOR["width"] - Settings.MONITOR["left"], Settings.MONITOR["height"] - Settings.MONITOR['top']))
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, (Settings.MONITOR["width"] - Settings.MONITOR["left"], Settings.MONITOR["height"] - Settings.MONITOR['top']))
        return warped

    def connect_img(self, img, imgFinal):
        alpha_s = imgFinal[:, :, 2] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            img[:img.shape[0], :img.shape[1], c] = (alpha_s * imgFinal[:, :, c] +
                                                    alpha_l * img[:img.shape[0], :img.shape[1], c])

        return img



    def draw_lanes(self, img, left_fit, right_fit):
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        color_img = np.zeros_like(img)

        left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
        right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])

        cv2.polylines(color_img, np.int32([left]), 0, (0, 0, 255), thickness=30)
        cv2.polylines(color_img, np.int32([right]), 0, (0, 0, 255), thickness=30)
        inv_perspective = self.inv_perspective_warp(color_img)

        return inv_perspective

    def get_curve(self, img, leftx, rightx):
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        y_eval = np.max(ploty)
        ym_per_pix = 1 / img.shape[0]  # meters per pixel in y dimension
        xm_per_pix = 1 / img.shape[1]  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

        #left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
         #   2 * left_fit_cr[0])
       # right_curverad = ((1 + (
         #           2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
          #  2 * right_fit_cr[0])


        car_pos = img.shape[1] / 2
        l_fit_x_int = left_fit_cr[0] * img.shape[0] ** 2 + left_fit_cr[1] * img.shape[0] + left_fit_cr[2]
        r_fit_x_int = right_fit_cr[0] * img.shape[0] ** 2 + right_fit_cr[1] * img.shape[0] + right_fit_cr[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
        center = (car_pos - lane_center_position) * xm_per_pix / 10
        # Now our radius of curvature is in meters

        return (l_fit_x_int, r_fit_x_int, center)#, (left_curverad, right_curverad)


    def thresholding(self, img):

        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kernel = np.ones((5, 5))
        imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 0)
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)


        imgCanny = cv2.Canny(imgBlur, 50, 150)

        mask = np.ones_like(imgCanny)*255
        cv2.fillPoly(mask, [Settings.VERTICES], 0)
        imgCanny = cv2.bitwise_and(mask, imgCanny)

        imgCanny = self.fill_gaps(imgCanny)


        return imgCanny

    def perspectivae_wrap(self, img):

        y = Settings.MONITOR["height"] - Settings.MONITOR['top']
        x = Settings.MONITOR["width"] - Settings.MONITOR['left']
        circles = Settings.SRC * np.array([x, y], np.float32)
        dst = Settings.DST * np.float32((x, y))

        M = cv2.getPerspectiveTransform(circles, dst)
        wraped = cv2.warpPerspective(img, M, (x, y))

        return wraped

    def draw_circle(self, img):
        circles = Settings.SRC
        circles = circles *np.float32((Settings.MONITOR["width"] - Settings.MONITOR['left'], Settings.MONITOR["height"]  - Settings.MONITOR['top']))
        img = cv2.circle(img, (circles[0][0], circles[0][1]), radius=5, color=(0, 0, 255), thickness=-1)
        img = cv2.circle(img, (circles[1][0], circles[1][1]), radius=5, color=(0, 0, 255), thickness=-1)
        img = cv2.circle(img, (circles[2][0], circles[2][1]), radius=5, color=(0, 0, 255), thickness=-1)
        img = cv2.circle(img, (circles[3][0], circles[3][1]), radius=5, color=(0, 0, 255), thickness=-1)

        return np.array(img)

    def get_hist(self, img):
        hist = np.sum(img[img.shape[0] // 2:, :], axis=0)
        return hist

    def sliding_window(self,img):

        img_out = np.zeros([img.shape[0], img.shape[1], 3])

        histogram = self.get_hist(img)
        midpoint = int(img.shape[1] / 2)
        left_point = np.argmax(histogram[:midpoint])
        right_point = np.argmax(histogram[midpoint:]) + midpoint

        hight_window = int(img.shape[0] / Settings.NWINDOWS)

        nozeros_matrix = np.nonzero(img)
        nozeros_x = np.array(nozeros_matrix[1])
        nozeros_y = np.array(nozeros_matrix[0])

        left_lane_inds = []
        right_lane_inds = []

        for i in range(Settings.NWINDOWS - 1):

            window_y_low = img.shape[0] - hight_window * i
            window_y_hight = img.shape[0] - hight_window * (i + 1)

            window_x_leftleft = left_point - Settings.MARGINES
            window_x_leftright = left_point + Settings.MARGINES

            window_x_rightleft = right_point - Settings.MARGINES
            window_x_rightright = right_point + Settings.MARGINES

            frame_nonzeros_left_ids = ((nozeros_y >= window_y_hight) & (nozeros_y < window_y_low)
                                       & (nozeros_x >= window_x_leftleft) & (nozeros_x < window_x_leftright)).nonzero()[0]

            frame_nonzeros_right_ids = ((nozeros_y >= window_y_hight) & (nozeros_y < window_y_low)
                                        & (nozeros_x >= window_x_rightleft) & (
                                                    nozeros_x < window_x_rightright)).nonzero()[0]

            left_lane_inds.append(frame_nonzeros_left_ids)
            right_lane_inds.append(frame_nonzeros_right_ids)

            if len(frame_nonzeros_left_ids) > Settings.MIN_PIX:
                left_point = int(np.mean(nozeros_x[frame_nonzeros_left_ids]))
            if len(frame_nonzeros_right_ids) > Settings.MIN_PIX:
                right_point = int(np.mean(nozeros_x[frame_nonzeros_right_ids]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nozeros_x[left_lane_inds]
        lefty = nozeros_y[left_lane_inds]
        rightx = nozeros_x[right_lane_inds]
        righty = nozeros_y[right_lane_inds]

        left_a = []
        left_b = []
        left_c = []

        right_a = []
        right_b = []
        right_c = []

        if leftx.size and rightx.size:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)

            ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])

            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

            img_out[nozeros_y[left_lane_inds], nozeros_x[left_lane_inds]] = [255, 0, 100]
            img_out[nozeros_y[right_lane_inds], nozeros_x[right_lane_inds]] = [0, 100, 255]

            return (left_fitx, right_fitx), (left_fit, right_fit), img_out
        else:
            return (0, 0), (0, 0), img
