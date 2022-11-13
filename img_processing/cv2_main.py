import datetime
import random
import shutil
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import imutils


class CV2Helper:
    def __init__(self):
        self.ws = 'E://nutrition5k_dataset'
        self.dataset_imagery_basedir = self.ws + '/imagery'
        self.dish_images_path = self.dataset_imagery_basedir + '/realsense_overhead/'
        self.dish_ids = ["dish_1557862829", "dish_1556572657", "dish_1562097001", "dish_1557861697", "dish_1557936599"]
        self.dish_folder = "dish_1557862829"
        self.rgb = '/rgb.png'
        self.depth_raw = '/depth_raw.png'
        self.depth_color = '/depth_color.png'
        # output_dir = ws + "/cv2_out"
        self.output_dir = self.ws + "/cv2_out"

    def read_img_and_rgb(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.copy()

    def view_thresholding(local_img, with_clahe=False):
        if with_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        local_img = clahe.apply(local_img)
        ret, thresh = cv2.threshold(local_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        plt.subplot(1, 2, 1)
        plt.imshow(local_img)
        plt.subplot(1, 2, 2)
        plt.imshow(thresh)
        plt.show()
        return thresh

    def apply_CLAHE(self, image, with_clahe=False):
        if with_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        copy_img = image.copy()
        c_img = clahe.apply(copy_img)
        ret, thresh = cv2.threshold(c_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        out = [c_img, thresh]
        return out

    def write_to_file(self, file_name, img):
        if not cv2.imwrite(file_name, img):
            print("Could not write image -> " + file_name)

    def write_contours(local_img, base_dir):
        contours, hierarchy = cv2.findContours(local_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print("Length of cont = ", len(contours))
        i = 0
        time_label = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = base_dir + time_label
        try:
            os.mkdir(output_dir)
            print("Create dir = " + base_dir)
        except:
            print("Create dir failed" + base_dir)

        for cnt in contours:
            # calculate epsilon base on contour's perimeter
            # contour's perimeter is returned by cv2.arcLength
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            # get approx polygons
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            # draw approx polygons
            tmp = local_img.copy()
            tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)
            tmp = cv2.drawContours(tmp, [approx], -1, (255, 0, 0), 5)
            f = output_dir + "/cont_" + str(i) + ".png"
            print("Saving to " + f)
            # write_to_file(f,tmp)
            i += 1
        return output_dir

    def step_1_read(self):
        loaded_imgs = []
        for d in self.dish_ids:
            dish_path = helper.dish_images_path + d + helper.rgb
            print(dish_path)
            dish_img = helper.read_img_and_rgb(dish_path)
            loaded_imgs.append(dish_img)
        return loaded_imgs

    def step_1_read(self, dish_paths):
        loaded_imgs = []
        for dish_path in dish_paths:
            print(dish_path)
            dish_img = helper.read_img_and_rgb(dish_path)
            loaded_imgs.append(dish_img)
        return loaded_imgs

    def step_2_apply_transformation(self, param_imgs):
        loaded_imgs = []
        clahe_ind = 0
        thresh_ind = 1
        for oi in param_imgs:
            img = oi.copy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # ret, thresh = cv2.threshold(img_processing, 127, 255, cv2.THRESH_BINARY)
            # out = self.apply_CLAHE(thresh,with_clahe=True)
            out = self.apply_CLAHE(img, with_clahe=True)
            # img_processing = cv2.Canny(out[thresh_ind], 50, 150)
            # img_processing = cv2.threshold(img_processing, 0, 255, cv2.THRESH_BINARY_INV)
            # kernel = np.ones((2,2),np.uint8)
            # img_processing = cv2.dilate(img_processing,  kernel,iterations=2)
            # img_processing = cv2.erode(img_processing, kernel,iterations=1)
            # out.append(img_processing)
            loaded_imgs.append(out)
        return loaded_imgs

    def apply_morph(self, img):
        img1 = img.copy()
        se1 = cv2.getStructuringElement(cv2.MORPH, (5, 5))
        se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        mask1 = cv2.morphologyEx(img1, cv2.MORPH_CLOSE, se1)
        mask2 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, se2)
        mask = np.dstack([mask1, mask2]) / 255
        out = img1 * mask
        return out

    def step3_apply_contour(self, image):
        min_area = 0.0005
        max_area = 0.095
        blur = 21
        dilate_iter = 10
        erode_iter = 10
        mask_color = 0.0
        # mask_color = (0.0, 0.0, 0.0)
        contour_info = [(c, cv2.contourArea(c)) for c in
                        cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]]
        # Get the area of the image as a comparison
        image_area = image.shape[0] * image.shape[1]
        # calculate max and min areas in terms of pixels
        max_ar = max_area * image_area
        min_ar = min_area * image_area
        # Set up mask with a matrix of 0's
        mask = np.zeros(image.shape, dtype=np.uint8)
        # Go through and find relevant contours and apply to mask
        for contour in contour_info:
            # Instead of worrying about all the smaller contours,
            # if the area is smaller than the min, the loop will break
            if min_ar < contour[1] < max_ar:
                # Add contour to mask
                mask = cv2.fillConvexPoly(mask, contour[0], (255))
                # use dilate, erode, and blur to smooth out the mask
                # mask = cv2.dilate(mask, None, iterations=dilate_iter)
                # mask = cv2.erode(mask, None, iterations=erode_iter)
                # mask = cv2.GaussianBlur(mask, (blur, blur), 0)

        # mask_stack = np.dstack([mask])
        # Ensures data types match up
        mask_stack = mask.astype('float32') / 255.0
        frame = image.astype('float32') / 255.0
        # Blend the image and the mask
        masked = (mask_stack * frame) + (mask_stack * mask_color)

        masked = (masked * 255).astype('uint8')

        return masked


helper = CV2Helper()
# orig_images = helper.step_1_read()
# out_images = helper.step_2_apply_transformation(orig_images)
# # print(len(out_images[0]))
# f, a = plt.subplots(4, 2)
# #
# # print(len(a))
# for i in range(0, len(orig_images[:2])):
#     a[0][i].imshow(orig_images[i])
#     clahe_img = out_images[i][0]
#     a[1][i].imshow(clahe_img)
#     thresh = out_images[i][1]
#     a[2][i].imshow(thresh)
#     canny = out_images[i][2]
#     a[3][i].imshow(canny)
#     # masked = helper.step3_apply_contour(canny)
#     # a[4][i].imshow(masked)
#
# plt.show()


w = 340
h = 340
iw = 640
ih = 480

helper = CV2Helper()
# orig_images = helper.step_1_read()
# # trial1 = orig_images[0].copy()
# # plt.imshow(trial1)
# # plt.show()
# print(trial1.shape)

# trial2 = cv2.resize(trial1,(340,340))
# plt.imshow(trial2)
# plt.show()
mask = np.zeros((ih, iw), np.uint8)
print(mask.shape)
# [30:460, 170:600]
# plt.imshow(trial1[80:380, 300:530])
# plt.show()
dishes = []
for d in os.listdir(helper.dish_images_path):
    dishes.append(helper.dish_images_path + d + helper.rgb)
# print(dishes)

samples = random.sample(dishes, 5)

orig_images = helper.step_1_read(samples)
out_images = helper.step_2_apply_transformation(orig_images)

f, a = plt.subplots(5, 6)
print(a[0])
for i in range(len(orig_images)):
    a[i][0].imshow(orig_images[i])
    a[i][0].set_title("Orig",fontsize=5)
    clahe_img = out_images[i][0]
    a[i][1].imshow(clahe_img)
    a[i][1].set_title("CLAHE",fontsize=5)
    thresh = out_images[i][1]
    a[i][2].imshow(thresh)
    a[i][1].set_title("THRESH",fontsize=5)
    canny = cv2.Canny(thresh, 20, 180)
    # a[i][3].imshow(canny)
    a[i][3].imshow(canny)
    a[i][3].set_title("CANNY",fontsize=5)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    drawing_img = np.ones_like(canny)
    drawing_img = cv2.drawContours(drawing_img, contours, -1, (0, 0, 0), 1)
    a[i][4].imshow(drawing_img)
    a[i][4].set_title("DRAW_CNT",fontsize=5)
    mask = np.zeros_like(drawing_img)
    bit_mask = cv2.bitwise_and(canny, canny, mask=mask)
    # a[i][5].set_title("BIT_MASK",fontsize=5)
    # a[i][5].imshow(bit_mask)

    g = cv2.cvtColor(orig_images[i],cv2.COLOR_RGB2GRAY)
    g = cv2.threshold(g,127,255,cv2.THRESH_BINARY_INV)
    a[i][5].set_title("BIT_MASK", fontsize=5)
    a[i][5].imshow(g)

plt.show()
