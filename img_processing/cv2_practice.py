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

ws = 'E://nutrition5k_dataset'
dataset_imagery_basedir = ws + '/imagery'
dish_images_path = dataset_imagery_basedir + '/realsense_overhead/'
dish_ids = ["dish_1557862829", "dish_1556572657", "dish_1562097001", "dish_1557861697", "dish_1557936599"]
dish_folder = "dish_1557862829"  # Looks good with OTSU with noise
# dish_folder = "dish_1556572657" # - Failed - square, better with OTSU with noise as it is a single image
# dish_folder = "dish_1562097001"  #- Works bcoz I had to do it manually ,Initial, still better with OTSU
# dish_folder = "dish_1557861697" #- Failed - round, still segments with OTSU
# dish_folder = "dish_1557936599"  # - Failed - rectangle - Works with OTSU
r = dish_images_path + dish_folder + '/rgb.png'
dr = dish_images_path + dish_folder + '/depth_raw.png'
dc = dish_images_path + dish_folder + '/depth_color.png'
# output_dir = ws + "/cv2_out"
output_dir = ws + "/cv2_out/clahe/"


def display_3dmask():
    img = cv2.imread(r)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # mask = cv2.imread(dr)
    mask = cv2.imread(dc)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    print(img.shape)
    plt.subplot(1, 2, 1)
    print(mask.shape)
    plt.imshow(mask)
    transparent = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
    transparent[:, :, 0:3] = img
    transparent[:, :, 3] = mask
    print(transparent.shape)
    plt.subplot(1, 2, 2)
    plt.imshow(transparent)
    plt.show()


def mask_rectangle(file, show_me=False):
    print("Read image " + file)
    if show_me:
        # plt.figure(figsize=(15, 9))
        img1 = cv2.imread(file)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        print(img1.shape)
        zero_mask = np.zeros(img1.shape[:2], dtype="uint8")
        print(zero_mask.shape)
        plt.subplot(1, 4, 1)
        plt.imshow(img1)

        plt.subplot(1, 4, 2)
        plt.imshow(zero_mask)

        rect_mask = cv2.rectangle(zero_mask, (170, 30), (600, 460), 255, -1)
        plt.subplot(1, 4, 3)
        plt.imshow(rect_mask)

        bit_mask = cv2.bitwise_and(img1, img1, mask=rect_mask)
        plt.subplot(1, 4, 4)
        plt.imshow(bit_mask)
        plt.show()
    else:
        img1 = cv2.imread(file)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        print(img1.shape)
        zero_mask = np.zeros(img1.shape[:2], dtype="uint8")
        print(zero_mask.shape)
        rect_mask = cv2.rectangle(zero_mask, (160, 30), (600, 460), 255, -1)
        # rect_mask = cv2.rectangle(zero_mask, (170, 30), (600, 460), 255, -1)
        bit_mask = cv2.bitwise_and(img1, img1, mask=rect_mask)
    return bit_mask


def show_max_contour(img):
    local_img = img.copy()
    ret, thresh = cv2.threshold(local_img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Length of cont = ", len(contours))
    contours = contours[0] if imutils.is_cv2() else contours[1]
    max_cont = max(contours, key=lambda x: cv2.contourArea(x))
    local_img = cv2.drawContours(local_img, max_cont, 0, (139, 0, 0), 5)
    return local_img


# noinspection DuplicatedCode
def display_text(local_img, approx):
    x = approx.ravel()[0]
    y = approx.ravel()[1] - 5
    if len(approx) == 3:
        cv2.putText(local_img, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
    elif len(approx) == 4:
        x1, y1, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        print(aspect_ratio)
        if 0.95 <= aspect_ratio <= 1.05:
            cv2.putText(local_img, "square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
        else:
            cv2.putText(local_img, "rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
    elif len(approx) == 5:
        cv2.putText(local_img, "Pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
    elif len(approx) == 10:
        cv2.putText(local_img, "Star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
    else:
        cv2.putText(local_img, "Unknown", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
    return local_img


def write_contours(local_img, path):
    ret, thresh = cv2.threshold(local_img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Length of cont = ", len(contours))
    i = 0
    try:
        shutil.rmtree(path)
        print("Cleanup" + path)
    except:
        print("Directory remove failed ")

    try:
        os.mkdir(path)
        print("Create dir = " + path)
    except:
        print("Create dir failed" + path)

    for cnt in contours:
        # calculate epsilon base on contour's perimeter
        # contour's perimeter is returned by cv2.arcLength
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        # get approx polygons
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # draw approx polygons

        # local_img = display_test(local_img, approx)
        # local_img = cv2.cvtColor(local_img, cv2.COLOR_GRAY2BGR)
        local_img = cv2.drawContours(local_img, [approx], -1, (255, 255, 255), 5)

        # hull is convex shape as a polygon
        # hull = cv2.convexHull(cnt)
        # local_img = cv2.drawContours(local_img, [hull], -1, (255, 0, 0), 5)
        # plt.imshow(local_img)
        # plt.show()
        # time.sleep(5)
        tmp = path + "/cont_" + str(i) + ".png"
        print("Saving to " + tmp)
        if not cv2.imwrite(tmp, local_img):
            print("Could not write image" + tmp)
        i += 1
    return output_dir


def show_contour_index(local_img, index):
    ret, thresh = cv2.threshold(local_img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Length of cont = ", len(contours))
    i = 0
    local_img = cv2.drawContours(local_img, contours, 65, (255, 255, 255), 5)
    t = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name = "{}/cont_index_{}.png".format(output_dir, str(index) + "_" + t)
    cv2.imwrite(file_name, local_img)
    print("File written to " + file_name)
    return local_img


def show_all_contours(local_img,last=False):
    local_img_copy = local_img.copy()
    # ret, thresh = cv2.threshold(local_img_copy, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Length of cont = ", len(contours))
    local_img_copy = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    if last :
        local_img_copy = cv2.drawContours(local_img_copy, contours, -1, (0, 0, 255), 3)
    else:
        local_img_copy = cv2.drawContours(local_img_copy, contours, len(contours)-1, (0, 0, 255), 3)
    return local_img_copy


def view_hist_eq(local_img):
    eq_img = cv2.equalizeHist(local_img)
    plt.subplot(2, 2, 1)
    plt.hist(local_img.flat, bins=100, range=(0, 255))
    plt.subplot(2, 2, 2)
    plt.hist(eq_img.flat, bins=100, range=(0, 255))
    plt.subplot(2, 2, 3)
    plt.imshow(local_img)
    plt.subplot(2, 2, 4)
    plt.imshow(eq_img)
    plt.show()


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


def read_img_and_rgb(p):
    img = cv2.imread(p)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


#
# plt.imshow(img_processing)
# plt.show()
# display_3dmask()

# Remove the extra space surronding the dish plate
# dp = dish_images_path + dish_folder + '/rgb.png'
# img_processing = read_img_and_rgb(dp)
# masked_img = mask_rectangle(dp)
# masked_rect = masked_img[30:460, 170:600]
# print("Masked Rectangle Shape", masked_rect.shape)
# plt.imshow(masked_rect)
# plt.show()
# End

# Show All Contours
# masked_rect_copy = masked_rect.copy()
# masked_rect_copy = cv2.cvtColor(masked_rect_copy, cv2.COLOR_RGB2GRAY)
# out = write_contours(masked_rect_copy)
# print("Files written to ", out)
# End

# View Histogram original image and second image
# hist_copy = masked_rect.copy()
# hist_copy = cv2.cvtColor(hist_copy, cv2.COLOR_RGB2GRAY)
# view_hist_eq(hist_copy)
# End

# View OTSU
# otsu_copy = masked_rect.copy()
# otsu_copy = cv2.cvtColor(otsu_copy, cv2.COLOR_RGB2GRAY)
# otsu_copy = view_thresholding(otsu_copy)
# write_contours(otsu_copy)
# End

# View OTSU + CLAHE
# for d in dish_ids:
#     dp = dish_images_path + d + '/rgb.png'
#     masked_img = mask_rectangle(dp)
#     masked_rect = masked_img[30:460, 170:600]
#     otsu_copy = masked_rect.copy()
#     otsu_copy = cv2.cvtColor(otsu_copy, cv2.COLOR_RGB2GRAY)
#     otsu_copy = view_thresholding(otsu_copy, with_clahe=True)
#     odir = output_dir + d
#     write_contours(otsu_copy, odir)
# End
path_arr = ["E://nutrition5k_dataset/cv2_out/clahe/dish_1557936599/cont_45.png",
            "E://nutrition5k_dataset/cv2_out/clahe/dish_1556572657/cont_32.png",
            "E://nutrition5k_dataset/cv2_out/clahe/dish_1557861697/cont_4.png",
            "E://nutrition5k_dataset/cv2_out/clahe/dish_1557862829/cont_8.png",
            "E:/nutrition5k_dataset/cv2_out/clahe/dish_1562097001/cont_0.png"]

plt.figure(figsize=(150, 90))
l = len(path_arr)
plt.subplots(l,2)
for i, p in enumerate(path_arr):
    # test = mask_rectangle(p,show_me=True)
    test = read_img_and_rgb(p)
    test = cv2.cvtColor(test, cv2.COLOR_RGB2GRAY)
    # test = test[30:460, 170:600]
    ret, thresh = cv2.threshold(test, 0, 255, cv2.THRESH_BINARY_INV)
    plt.subplot(i+1,3,1)
    plt.imshow(test)
    plt.subplot(i+1,3,2)
    plt.imshow(thresh)
    # plt.subplot(i+1,3,3)
    # plt.imshow(show_all_contours(thresh))
    # write_contours(thresh, output_dir + "v2")
    plt.show()

# End
# Show Index
# indexed_contour = show_contour_index(masked_rect_copy, 65)
# plt.imshow(indexed_contour)
# plt.show()

# masked_rect_copy = masked_rect.copy()
# masked_rect_copy = cv2.cvtColor(masked_rect_copy, cv2.COLOR_BGR2GRAY)
# # plt.subplot(1, 3, 1)
# print("Printing masked_rect_copy")
# plt.imshow(masked_rect_copy)
# plt.show()
#
#
# # show_contours(masked_rect_copy)
# # masked_rect_copy_1 = show_all_contours(masked_rect_copy)
#
# masked_rect_copy_1 = show_max_contour(masked_rect_copy)
#
# # masked_canny = masked_rect_copy.copy()
#
# # masked_canny = cv2.Canny(masked_canny, 90, 100)
#
# plt.imshow(masked_rect_copy_1)
# plt.show()
