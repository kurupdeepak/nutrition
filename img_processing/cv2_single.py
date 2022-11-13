import numpy as np
import random
import os
import cv2
import matplotlib.pyplot as plt


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


helper = CV2Helper()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

dishes = []
for d in os.listdir(helper.dish_images_path):
    dishes.append(helper.dish_images_path + d + helper.rgb)

img_index = random.randint(0, len(dishes))
print("Going to load index ", img_index)


def show_img(r, c, index, img, label):
    plt.subplot(r, c, index)
    plt.title(label)
    plt.imshow(img)


class ImageInfo:
    def __init__(self, img, label):
        self.image = img
        self.label = label


def basic(p_img):
    img = cv2.imread(p_img)
    print("Image = " + dishes[img_index])

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

    clahe_img = clahe.apply(gray)

    ret, otsu_clahe = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ret, otsu_gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [ImageInfo(img, "bgr"),
            ImageInfo(rgb, "rgb"),
            ImageInfo(gray, "gray"),
            ImageInfo(clahe_img, "clahe"),
            ImageInfo(otsu_clahe, "otsu-clahe"),
            ImageInfo(otsu_gray, "otsu-gray")]


def draw_contours(img, orig, out):
    img_copy = img.copy();
    orig_copy = orig.copy()
    cv2.imwrite(out + "/orig.png", orig)
    # contours, hierarchy = cv2.findContours(img_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # # for i, c in enumerate(contours):
    # draw = cv2.drawContours(orig_copy, contours, 0, 255, 5)
    # cv2.imwrite(out + "/0.png", draw)
    # # cv2.imwrite(out + "/0" + str(i) + ".png", draw)


# def main():
#     # check basic image variations
#     # dish = dishes[img_index]
#     dish = "E://nutrition5k_dataset/imagery/realsense_overhead/dish_1561480439/rgb.png"
#     basic_img = basic(dish)
#     r = 2
#     c = 3
#     f, a = plt.subplots(r, c)
#     for i, img_info in enumerate(basic_img):
#         show_img(r, c, i + 1, img_info.image, img_info.label)
#     plt.show()
#     otsu_gray = basic_img[5].image
#     rgb_img = basic_img[1].image
#     draw_contours(otsu_gray, rgb_img, helper.output_dir)
#     print("Contours written to " + helper.output_dir)

def main():
    n = 115
    iters = 0
    while n % 3 != 0:
        n = n + 1
        iters += 1
    print(n)
    print(iters)
    np.arange(115).reshape()


main()
