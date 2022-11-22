import pandas as pd
import os
import matplotlib.pyplot as plt


class DatasetUtil:
    @staticmethod
    def get_top_30_ingredients(dish_ingredients):
        dish_ing_by_count = dish_ingredients[['dish_id', 'name', 'id']] \
            .groupby('dish_id', as_index=False).id.count() \
            .sort_values(by="id", ascending=False)
        return dish_ing_by_count

    @staticmethod
    def check_dir(dish_ids, image_dir):
        # print("Dish Shape",dish_ids.shape)
        df = pd.DataFrame(columns=['dish_id', 'exists'])
        for dish_id in dish_ids:
            # print(dish_images +  id)
            df.loc[len(df.index)] = [dish_id, os.path.exists(image_dir + dish_id)]
        return df

    @staticmethod
    def get_image_path(dish_ids,
                       image_dir,
                       file_name="/rgb.png"):
        # print("Dish Shape",dish_ids)
        images = []
        for dish_id in dish_ids:
            if os.path.exists(image_dir + dish_id):
                images.append({"dish_id": dish_id, "image_path": image_dir + dish_id + file_name})
        df = pd.DataFrame(images, columns=["dish_id", "image_path"])
        return df

    @staticmethod
    def get_rgb_image(dish_ids,
                      image_dir,
                      file_name="/rgb.png"):
        # print("Dish Shape",dish_ids)
        images = []
        for dish_id in dish_ids:
            # print("Types = ",type(dish_images_path),type(dish_id))
            if os.path.exists(image_dir + dish_id):
                images.append({"dish_id": dish_id, "image": plt.imread(image_dir + dish_id + file_name)})
        return images
