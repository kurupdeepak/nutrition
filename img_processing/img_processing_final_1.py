import numpy as np
import matplotlib.pyplot as plt
import cv2
import random


def crop(img):
    # argwhere will give you the coordinates of every non-zero point
    true_points = np.argwhere(img)
    # take the smallest points and use them as the top left of your crop
    top_left = true_points.min(axis=0)
    # take the largest points and use them as the bottom right of your crop
    bottom_right = true_points.max(axis=0)
    out = img[top_left[0]:bottom_right[0] + 1,  # plus 1 because slice isn't
          top_left[1]:bottom_right[1] + 1]  # inclusive
    return out


def rectangular_mask(dish_path,n=2):
    g = 1
    c = 3
    plt.figure(figsize=(12, 10))
    # for dish_path in dishes:
    orig = cv2.imread(dish_path)
    image = orig.copy()
    orig = cv2.cvtColor(orig,cv2.COLOR_BGR2RGB)
    # print(image.shape)
    plt.subplot(n, c, g)
    plt.imshow(image)
    mask = np.zeros(image.shape[:2], dtype="uint8")
    # mask = cv2.rectangle(mask, (0, 480), (180, 580), 255, -1)
    mask = cv2.rectangle(mask, (180, 30), (600, 480), 255, -1)
    g = g + 1
    plt.subplot(n, c, g)
    plt.imshow(mask)
    # apply our mask -- notice how only the person in the image is
    # cropped out
    g = g + 1
    masked = cv2.bitwise_or(image, image, mask=mask)
    masked = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
    plt.subplot(n, c, g)
    plt.imshow(masked)
    g += 1
    return orig,masked


def remove_blue(hsv_image):
    # input_image = tmp.copy()
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

def test(n=2):
  index = random.sample(list(np.arange(len(dishes))),n)
  img_paths = [dishes[i] for i in index]
  output = []
  for i,path in enumerate(img_paths):
    orig,output = rectangular_mask(path,n)
    cropped_out = crop(output)
    hsv = cv2.cvtColor(cropped_out, cv2.COLOR_BGR2HSV)
    imgfinal = remove_blue(hsv)
    copy_of = imgfinal.copy()
    copy_of[copy_of > 0] = 255
    final_img = cv2.bitwise_and(orig[i],orig[i],mask=copy_of)
    output.append([orig,final_img])
  return output