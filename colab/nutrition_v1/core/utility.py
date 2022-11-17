import matplotlib.pyplot as plt
import pandas as pd
from core.common import dish_images_path
import os

def load_dish_metadata(filepath):
    f = open(filepath, 'r')
    lines = f.readlines()
    dish_cols = ["dish_id", "total_calories", "total_mass", "total_fat", "total_carb", "total_protein"]
    ing_cols = ["dish_id", "id", "name", "grams", "calories", "fat", "carb", "protein"]

    dish_info = []
    dish_ingredients = []
    s = 6
    step = 7
    line_num = 1
    for line in lines:
        # print(line)
        dish_line = line.split(',')
        dish = dish_line[:s]
        # print(line_num,dish)
        ing_len = len(dish_line[s:])
        tkn = s
        dish_info.append(pd.Series(dish, index=dish_cols))
        # print("Going for ingredients",tkn,ing_len)
        while tkn < ing_len:
            row = [dish[0]]
            for c in dish_line[tkn:tkn + step]:
                row.append(c.strip())
            # print(row)
            dish_ingredients.append(pd.Series(row, index=ing_cols))
            tkn += step
        line_num += 1
        # break
    d = pd.DataFrame(dish_info, columns=dish_cols)
    di = pd.DataFrame(dish_ingredients, columns=ing_cols)
    return d, di


def check_dir(dish_ids):
    # print("Dish Shape",dish_ids.shape)
    df = pd.DataFrame(columns=['dish_id', 'exists'])
    for dish_id in dish_ids:
        # print(dish_images +  id)
        df.loc[len(df.index)] = [dish_id, os.path.exists(dish_images_path + dish_id)]
    return df


def get_rgb_imagepath(dish_ids):
    # print("Dish Shape",dish_ids)
    images = []
    for dish_id in dish_ids:
        if os.path.exists(dish_images_path + dish_id):
            images.append({"dish_id": dish_id, "image_path": dish_images_path + dish_id + '/rgb.png'})
    df = pd.DataFrame(images, columns=["dish_id", "image_path"])
    return df


def get_rgb_image(dish_ids):
    # print("Dish Shape",dish_ids)
    images = []
    for dish_id in dish_ids:
        # print("Types = ",type(dish_images_path),type(dish_id))
        if os.path.exists(dish_images_path + dish_id):
            images.append({"dish_id": dish_id, "image": plt.imread(dish_images_path + dish_id + '/rgb.png')})
    return images


# If you wanted to, you could really turn this into a helper function to load in with a helper.py script...
# Plot the validation and training data separately
def plot_loss_curves(history):
    """
    Returns separate loss curves for training and validation metrics.
    """
    plotAccuracy = False

    if 'accuracy' in history.history.keys():
        plotAccuracy = True

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(history.history['loss']))

    # Plot loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # # Plot accuracy
    if plotAccuracy:
        # history.history['accuracy']
        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        plt.figure()
        plt.plot(epochs, accuracy, label='training_accuracy')
        plt.plot(epochs, val_accuracy, label='val_accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.legend()
