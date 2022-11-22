import matplotlib.pyplot as plt
import pandas as pd
import random

from core.data import DatasetConfig, DatasetUtil
from core.data.dish_info import DishInfo


class DataLoader:
    def __init__(self, data_config: DatasetConfig,
                 debug: bool = False):
        self.rgb_train = None
        self.rgb_test = None
        self.depth_test = None
        self.depth_train = None
        self.test_dish_ids = None
        self.train_dish_ids = None
        self.data_config = data_config
        self.cafe1_dish_info = DishInfo(data_config.metadata_dir + data_config.dish_cafe1_file)
        self.cafe2_dish_info = DishInfo(data_config.metadata_dir + data_config.dish_cafe2_file)

        cafe_1, cafe1_ing = self.cafe1_dish_info.get_dish_info()
        cafe_2, cafe2_ing = self.cafe2_dish_info.get_dish_info()
        self.LOG_HANDLE = "DataLoader -->"
        if debug:  # Display Information on the loaded files
            print(self.LOG_HANDLE, "Cafe = 1", "*" * 50)
            print(self.LOG_HANDLE, cafe_1.info())
            print(self.LOG_HANDLE, cafe_1.shape)
            print(self.LOG_HANDLE, "Cafe = 1 Dish Ingredients", "*" * 50)
            print(self.LOG_HANDLE, cafe1_ing.info())
            print(self.LOG_HANDLE, cafe1_ing.shape)

            print(self.LOG_HANDLE, "Cafe = 2", "*" * 50)
            print(self.LOG_HANDLE, cafe_2.info())
            print(self.LOG_HANDLE, "Shape = ", cafe_2.shape)
            print(self.LOG_HANDLE, "Cafe = 2 Dish Ingredients", "*" * 50)
            print(self.LOG_HANDLE, cafe2_ing.info())
            print(self.LOG_HANDLE, "Shape = ", cafe2_ing.shape)

        self.dish = pd.concat([cafe_1, cafe_2])
        self.dish_ingredients = pd.concat([cafe1_ing, cafe2_ing])
        print(self.LOG_HANDLE, "Total Dishes", self.dish.shape)
        print(self.LOG_HANDLE, "Total Dish Ingredients", self.dish_ingredients.shape)

        self.__cast_astype_float()

        self.__split_data(debug)

        self.__verify_images(False)

    def get_dishes(self):
        return self.dish

    def get_dish_ingredients(self):
        return self.dish_ingredients

    def get_splits(self):
        return self.train_dish_ids, self.test_dish_ids

    def __split_data(self, debug=False):
        self.depth_train = pd.read_csv(self.data_config.splits_dir + self.data_config.depth_train_file,
                                       header=None,
                                       names=self.data_config.dish_id_col)

        self.depth_test = pd.read_csv(self.data_config.splits_dir + self.data_config.depth_test_file,
                                      header=None,
                                      names=self.data_config.dish_id_col)

        self.rgb_test = pd.read_csv(self.data_config.splits_dir + self.data_config.rgb_test_file,
                                    header=None,
                                    names=self.data_config.dish_id_col)
        self.rgb_train = pd.read_csv(self.data_config.splits_dir + self.data_config.rgb_train_file,
                                     header=None,
                                     names=self.data_config.dish_id_col)
        if debug:
            print(self.LOG_HANDLE, "Depth train split ids shape = ", self.depth_train.shape)
            print(self.LOG_HANDLE, self.depth_train.head())
            print(self.LOG_HANDLE, "RGB train split ids shape = ", self.rgb_train.shape)
            print(self.LOG_HANDLE, self.rgb_train.head())
            print(self.LOG_HANDLE, "Depth test split ids shape = ", self.depth_test.shape)
            print(self.LOG_HANDLE, self.depth_test.head())
            print(self.LOG_HANDLE, "RGB test split ids shape = ", self.rgb_test.shape)
            print(self.LOG_HANDLE, self.rgb_test.head())

        self.train_dish_ids = pd.merge(self.depth_train, self.rgb_train)
        self.test_dish_ids = pd.merge(self.depth_test, self.rgb_test)
        if debug:
            print(self.LOG_HANDLE, "Train Dish Ids = ", self.train_dish_ids.shape)
            print(self.LOG_HANDLE, "Test Dish Ids = ", self.test_dish_ids.shape)

    def __verify_images(self, debug):
        s1 = DatasetUtil.check_dir(self.test_dish_ids.dish_id, self.data_config.image_dir)
        s2 = DatasetUtil.check_dir(self.train_dish_ids.dish_id, self.data_config.image_dir)
        s3 = DatasetUtil.check_dir(self.rgb_train.dish_id, self.data_config.image_dir)
        s4 = DatasetUtil.check_dir(self.rgb_test.dish_id, self.data_config.image_dir)

        if debug:
            print(self.LOG_HANDLE, "Test Dish Ids = ", self.test_dish_ids.shape, s1.shape,
                  s1[s1.exists is True].shape)
            print(self.LOG_HANDLE, "Train Dish Ids = ", self.train_dish_ids.shape, s2.shape,
                  s2[s2.exists is True].shape)
            print(self.LOG_HANDLE, "RGB Train Ids = ", self.rgb_train.shape, s3.shape,
                  s3[s3.exists is True].shape)
            print(self.LOG_HANDLE, "RGB Test Ids = ", self.rgb_test.shape, s4.shape,
                  s4[s4.exists is True].shape)
            print(self.LOG_HANDLE, "RGB Train ", s3.exists.value_counts())
            print(self.LOG_HANDLE, "RGB Test ", s4.exists.value_counts())

            i = random.randrange(0, len(self.train_dish_ids))
            # print(self.LOG_HANDLE,i,train_dish_ids.dish_id[i])
            img_path = self.data_config.image_dir + self.train_dish_ids.dish_id[i]
            # print(self.LOG_HANDLE,img_path)
            f, a = plt.subplots(1, 3)

            a[0].imshow(plt.imread(img_path + '/depth_color.png'))
            a[1].imshow(plt.imread(img_path + '/depth_raw.png'))
            a[2].imshow(plt.imread(img_path + '/rgb.png'))

            plt.show()

            i_test = random.randrange(0, len(self.test_dish_ids))
            # print(self.LOG_HANDLE,i,train_dish_ids.dish_id[i])
            img_path = self.data_config.image_dir + self.test_dish_ids.dish_id[i_test]
            # print(self.LOG_HANDLE,img_path)
            f, a = plt.subplots(1, 3)

            a[0].imshow(plt.imread(img_path + '/depth_color.png'))
            a[1].imshow(plt.imread(img_path + '/depth_raw.png'))
            a[2].imshow(plt.imread(img_path + '/rgb.png'))

            plt.show()

        print(self.LOG_HANDLE, "Dish Master", self.dish.shape)
        print(self.LOG_HANDLE, "Dish Ingredients Master", self.dish_ingredients.shape)
        print(self.LOG_HANDLE, "Training Dish Ids", self.train_dish_ids.shape)
        print(self.LOG_HANDLE, "Test Dish Ids", self.test_dish_ids.shape)

    def __cast_astype_float(self):
        self.dish_ingredients['grams'] = self.dish_ingredients.grams.astype("float32")
        self.dish_ingredients['calories'] = self.dish_ingredients.calories.astype("float32")
        self.dish_ingredients['fat'] = self.dish_ingredients.fat.astype("float32")
        self.dish_ingredients['carb'] = self.dish_ingredients.carb.astype("float32")
        self.dish_ingredients['protein'] = self.dish_ingredients.protein.astype("float32")
        self.dish["total_calories"] = self.dish.total_calories.astype("float32")
        self.dish["total_mass"] = self.dish.total_mass.astype("float32")
        self.dish["total_fat"] = self.dish.total_fat.astype("float32")
        self.dish["total_carb"] = self.dish.total_carb.astype("float32")
        self.dish["total_protein"] = self.dish.total_protein.astype("float32")
