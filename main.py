from init_data import load_available_data, normalize_numeric_data

FILE_TAG = "main-py : "
print(FILE_TAG, "Start loading data ")

dish, dish_ingredients = load_available_data()

print(dish.info())
print(dish_ingredients.info())

print ("Normalize numeric data fields")

normalize_numeric_data(dish,dish_ingredients)

