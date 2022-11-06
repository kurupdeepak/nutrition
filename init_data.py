# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import random

import pandas as pd
from matplotlib import pyplot as plt

from core.settings import *
from core.utility import load_dish_metadata, check_dir, get_rgb_imagepath

FILE_TAG = "init_data_py : "


def load_available_data():
    dish, dish_ingredients, test_ids, train_ids = get_dish_data()

    print(FILE_TAG, "Dish Information = ", dish.shape)
    print(FILE_TAG, "Dish Ingredient Information = ", dish_ingredients.shape)

    print(FILE_TAG, "Dish data top rows")
    print(dish.head())
    print(FILE_TAG, "Dish Ingredients data top rows")

    print(dish_ingredients.head())

    print(dish.info())

    print("-----" * 30)

    cast_to_float(dish, dish_ingredients)

    print(dish_ingredients.head())

    print(dish.info())
    print("-----" * 30)

    dish_images = get_rgb_imagepath(dish.dish_id)

    merged_ids = pd.concat([train_ids.dish_id, test_ids.dish_id])

    dish_data = dish[dish.dish_id.isin(merged_ids)]
    dish_ing_data = dish_ingredients[dish_ingredients.dish_id.isin(merged_ids)]

    dish_data = dish_images.merge(dish_data)
    return dish_data, dish_ing_data


def get_dish_data(debug: bool = False):
    cafe_1, cafe1_ing = load_dish_metadata(dish_metadata_basedir + dish_cafe1_file)
    cafe_2, cafe2_ing = load_dish_metadata(dish_metadata_basedir + dish_cafe2_file)

    if debug:  # Display Information on the loaded files
        print(FILE_TAG, "Cafe = 1", "*" * 50)
        print(FILE_TAG, cafe_1.info())
        print(FILE_TAG, cafe_1.shape)
        print(FILE_TAG, "Cafe = 1 Dish Ingredients", "*" * 50)
        print(FILE_TAG, cafe1_ing.info())
        print(FILE_TAG, cafe1_ing.shape)

        print(FILE_TAG, "Cafe = 2", "*" * 50)
        print(FILE_TAG, cafe_2.info())
        print(FILE_TAG, "Shape = ", cafe_2.shape)
        print(FILE_TAG, "Cafe = 2 Dish Ingredients", "*" * 50)
        print(FILE_TAG, cafe2_ing.info())
        print(FILE_TAG, "Shape = ", cafe2_ing.shape)

    dish = pd.concat([cafe_1, cafe_2])
    dish_ingredients = pd.concat([cafe1_ing, cafe2_ing])
    print(FILE_TAG, "Total Dishes", dish.shape)
    print(FILE_TAG, "Total Dish Ingredients", dish_ingredients.shape)

    depth_train = pd.read_csv(splits_dir + depth_train_file, header=None, names=dish_id_col)
    depth_test = pd.read_csv(splits_dir + depth_test_file, header=None, names=dish_id_col)
    rgb_test = pd.read_csv(splits_dir + rgb_test_file, header=None, names=dish_id_col)
    rgb_train = pd.read_csv(splits_dir + rgb_train_file, header=None, names=dish_id_col)

    if debug:
        print(FILE_TAG, "Depth train split ids shape = ", depth_train.shape)
        print(FILE_TAG, depth_train.head())
        print(FILE_TAG, "RGB train split ids shape = ", rgb_train.shape)
        print(FILE_TAG, rgb_train.head())
        print(FILE_TAG, "Depth test split ids shape = ", depth_test.shape)
        print(FILE_TAG, depth_test.head())
        print(FILE_TAG, "RGB test split ids shape = ", rgb_test.shape)
        print(FILE_TAG, rgb_test.head())

    train_dish_ids = pd.merge(depth_train, rgb_train)
    test_dish_ids = pd.merge(depth_test, rgb_test)
    if debug:
        print(FILE_TAG, "Train Dish Ids = ", train_dish_ids.shape)
        print(FILE_TAG, "Test Dish Ids = ", test_dish_ids.shape)

    s1 = check_dir(test_dish_ids.dish_id)
    s2 = check_dir(train_dish_ids.dish_id)
    s3 = check_dir(rgb_train.dish_id)
    s4 = check_dir(rgb_test.dish_id)
    if debug:
        print(FILE_TAG, "Test Dish Ids = ", test_dish_ids.shape, s1.shape, s1[s1.exists == True].shape)
        print(FILE_TAG, "Train Dish Ids = ", train_dish_ids.shape, s2.shape, s2[s2.exists == True].shape)
        print(FILE_TAG, "RGB Train Ids = ", rgb_train.shape, s3.shape, s3[s3.exists == True].shape)
        print(FILE_TAG, "RGB Test Ids = ", rgb_test.shape, s4.shape, s4[s4.exists == True].shape)
        print(FILE_TAG, "RGB Train ", s3.exists.value_counts())
        print(FILE_TAG, "RGB Test ", s4.exists.value_counts())

        i = random.randrange(0, len(train_dish_ids))
        # print(FILE_TAG,i,train_dish_ids.dish_id[i])
        img_path = dish_images_path + train_dish_ids.dish_id[i]
        # print(FILE_TAG,img_path)
        f, a = plt.subplots(1, 3)

        a[0].imshow(plt.imread(img_path + '/depth_color.png'))
        a[1].imshow(plt.imread(img_path + '/depth_raw.png'))
        a[2].imshow(plt.imread(img_path + '/rgb.png'))

        plt.show()

        i_test = random.randrange(0, len(test_dish_ids))
        # print(FILE_TAG,i,train_dish_ids.dish_id[i])
        img_path = dish_images_path + test_dish_ids.dish_id[i_test]
        # print(FILE_TAG,img_path)
        f, a = plt.subplots(1, 3)

        a[0].imshow(plt.imread(img_path + '/depth_color.png'))
        a[1].imshow(plt.imread(img_path + '/depth_raw.png'))
        a[2].imshow(plt.imread(img_path + '/rgb.png'))

        plt.show()

    print(FILE_TAG, "Dish Master", dish.shape)
    print(FILE_TAG, "Dish Ingredients Master", dish_ingredients.shape)
    print(FILE_TAG, "Training Dish Ids", train_dish_ids.shape)
    print(FILE_TAG, "Test Dish Ids", test_dish_ids.shape)

    if debug:
        d1 = train_dish_ids.query('dish_id in @dish.dish_id')
        print(FILE_TAG, d1.shape)
        d1 = train_dish_ids.query('dish_id in @dish_ingredients.dish_id')
        print(FILE_TAG, d1.shape)
        d1 = test_dish_ids.query('dish_id in @dish.dish_id')
        print(FILE_TAG, d1.shape)
        d1 = test_dish_ids.query('dish_id in @dish_ingredients.dish_id')
        print(FILE_TAG, d1.shape)

        print(FILE_TAG, dish.info())

        print(FILE_TAG, dish_ingredients.info())

    return dish, dish_ingredients, test_dish_ids, train_dish_ids


def get_top_30_ingredients(dish_ingredients):
    dish_ing_by_count = dish_ingredients[['dish_id', 'name', 'id']] \
        .groupby('dish_id', as_index=False).id.count() \
        .sort_values(by="id", ascending=False)
    return dish_ing_by_count


def normalize_bymax(df, column):
    max_val = df[column].max()
    fld_name = column + "_norm"
    df[fld_name] = df[column] / max_val


def cast_to_float(dish, dish_ingredients):
    dish_ingredients['grams'] = dish_ingredients.grams.astype("float32")
    dish_ingredients['calories'] = dish_ingredients.calories.astype("float32")
    dish_ingredients['fat'] = dish_ingredients.fat.astype("float32")
    dish_ingredients['carb'] = dish_ingredients.carb.astype("float32")
    dish_ingredients['protein'] = dish_ingredients.protein.astype("float32")
    dish["total_calories"] = dish.total_calories.astype("float32")
    dish["total_mass"] = dish.total_mass.astype("float32")
    dish["total_fat"] = dish.total_fat.astype("float32")
    dish["total_carb"] = dish.total_carb.astype("float32")
    dish["total_protein"] = dish.total_protein.astype("float32")


def normalize_numeric_data(dish, dish_ingredients):
    normalize_bymax(dish, "total_calories")
    normalize_bymax(dish, "total_mass")
    normalize_bymax(dish, "total_fat")
    normalize_bymax(dish, "total_carb")
    normalize_bymax(dish, "total_protein")
    normalize_bymax(dish_ingredients, "grams")
    normalize_bymax(dish_ingredients, "calories")
    normalize_bymax(dish_ingredients, "fat")
    normalize_bymax(dish_ingredients, "carb")
    normalize_bymax(dish_ingredients, "protein")
