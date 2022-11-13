import numpy as np
import matplotlib.pyplot as plt
import cv2
import random


class ImagePreProcessor:
    def __init__(self, dishes):
        self.dishes = dishes

    def crop(self, img):
        # argwhere will give you the coordinates of every non-zero point
        true_points = np.argwhere(img)
        # take the smallest points and use them as the top left of your crop
        top_left = true_points.min(axis=0)
        # take the largest points and use them as the bottom right of your crop
        bottom_right = true_points.max(axis=0)
        out = img[top_left[0]:bottom_right[0] + 1,  # plus 1 because slice isn't
              top_left[1]:bottom_right[1] + 1]  # inclusive
        return out

    def rectangular_mask(self, dish_path):
        image_bgr = cv2.imread(dish_path)
        mask = np.zeros(image_bgr.shape[:2], dtype="uint8")
        # mask = cv2.rectangle(mask, (0, 480), (180, 580), 255, -1)
        mask = cv2.rectangle(mask, (180, 30), (600, 480), 255, -1)
        masked = cv2.bitwise_or(image_bgr, image_bgr, mask=mask)
        masked = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
        return image_bgr, mask, masked

    def remove_blue(self, hsv_image):
        input_image = hsv_image.copy()

        grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        # Convert the BGR image to HSV:
        hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

        # Create the HSV range for the blue ink:
        # [128, 255, 255], [90, 50, 70]
        lower_values = np.array([90, 50, 70])
        upper_values = np.array([128, 255, 255])

        # Get binary mask of the blue ink:
        blue_mask = cv2.inRange(hsv_image, lower_values, upper_values)
        # Use a little bit of morphology to clean the mask:
        # Set kernel (structuring element) size:
        kernel_size = 3
        # Set morph operation iterations:
        op_iterations = 1
        # Get the structuring element:
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        # Perform closing:
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, morph_kernel, None, None, op_iterations,
                                     cv2.BORDER_REFLECT101)

        # Add the white mask to the grayscale image:
        color_mask = cv2.add(grayscale_image, blue_mask)
        _, binary_image = cv2.threshold(color_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # cv2.imwrite('bwimage.jpg',binary_image)
        thresh, im_bw = cv2.threshold(binary_image, 210, 230, cv2.THRESH_BINARY)
        kernel = np.ones((1, 1), np.uint8)
        image_final = cv2.dilate(im_bw, kernel=kernel, iterations=1)
        return image_final

    def get_final_img(self, image_path, crop_only=False):
        orig_bgr, mask, masked = self.rectangular_mask(image_path)
        cropped_out = self.crop(masked)
        out = []
        if crop_only:
            out.append([orig_bgr, mask, masked, cropped_out, None])
        else:
            rgb = cropped_out.copy()
            hsv = cv2.cvtColor(cropped_out, cv2.COLOR_BGR2HSV)
            imgfinal = self.remove_blue(hsv)
            copy_of = imgfinal.copy()
            # copy_of[copy_of > 0] = 255
            final_img = cv2.bitwise_and(rgb, rgb, mask=copy_of)
            out.append([orig_bgr, mask, masked, cropped_out, final_img])
        return out

    def test(self, n=2, crop_only=False):
        index = random.sample(list(np.arange(len(self.dishes))), n)
        img_paths = [self.dishes[i] for i in index]
        out = []
        for i, path in enumerate(img_paths):
            orig_bgr, mask, masked = self.rectangular_mask(path)
            cropped_out = self.crop(masked)
            if crop_only:
                out.append([orig_bgr, mask, masked, cropped_out, None])
            else:
                rgb = cropped_out.copy()
                hsv = cv2.cvtColor(cropped_out, cv2.COLOR_BGR2HSV)
                imgfinal = self.remove_blue(hsv)
                copy_of = imgfinal.copy()
                # copy_of[copy_of > 0] = 255
                final_img = cv2.bitwise_and(rgb, rgb, mask=copy_of)
                out.append([orig_bgr, mask, masked, cropped_out, final_img])
        return out

    def display_sample(self, sample, crop_only=False):
        grid = 1
        n = len(sample)
        c = 4 if crop_only else 5
        plt.figure(figsize=(15, 10))
        for i, lst in enumerate(sample):
            img = cv2.cvtColor(lst[0], cv2.COLOR_BGR2RGB)
            plt.subplot(n, c, grid)
            plt.imshow(img)
            plt.title("original")

            grid += 1
            plt.subplot(n, c, grid)
            plt.imshow(lst[1])
            plt.title("mask")

            grid += 1
            plt.subplot(n, c, grid)
            plt.imshow(lst[2])
            plt.title("masked-out")

            grid += 1
            plt.subplot(n, c, grid)
            plt.imshow(lst[3])
            plt.title("cropped")

            if not crop_only:
                grid += 1
                plt.subplot(n, c, grid)
                plt.imshow(lst[4])
                plt.title("final-out")
            grid += 1
