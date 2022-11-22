import pandas as pd
import os


class DishInfo:
    def __init__(self, filepath):
        self.file_path = filepath
        lines = self.read_file()
        self.dish_cols = ["dish_id",
                          "total_calories",
                          "total_mass",
                          "total_fat",
                          "total_carb",
                          "total_protein"]

        self.ingredients_col = ["dish_id",
                                "id",
                                "name",
                                "grams",
                                "calories",
                                "fat",
                                "carb",
                                "protein"]
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
            dish_info.append(pd.Series(dish, index=self.dish_cols))
            # print("Going for ingredients",tkn,ing_len)
            while tkn < ing_len:
                row = [dish[0]]
                for c in dish_line[tkn:tkn + step]:
                    row.append(c.strip())
                # print(row)
                dish_ingredients.append(pd.Series(row, index=self.ingredients_col))
                tkn += step
            line_num += 1
            # break
        self.dishes = pd.DataFrame(dish_info, columns=self.dish_cols)
        self.dish_ingredients = pd.DataFrame(dish_ingredients, columns=self.ingredients_col)

    def read_file(self):
        f = open(self.file_path, 'r')
        lines = f.readlines()
        return lines

    def get_dishes(self):
        return self.dishes

    def get_dish_ingredients(self):
        return self.dish_ingredients

    def get_dish_info(self):
        return self.dishes, self.dish_ingredients
