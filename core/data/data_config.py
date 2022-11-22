class DatasetConfig:
    def __init__(self,
                 base_dir=None,
                 image_dir=None,
                 metadata_dir=None,
                 splits_dir=None):
        self.workspace = base_dir
        self.image_dir = self.workspace + image_dir
        self.metadata_dir = self.workspace + metadata_dir
        self.splits_dir = self.workspace + splits_dir
        self.depth_train_file = "/depth_train_ids.txt"
        self.depth_test_file = "/depth_test_ids.txt"
        self.rgb_train_file = "/rgb_train_ids.txt"
        self.rgb_test_file = "/rgb_test_ids.txt"
        self.dish_cafe1_file = "/dish_metadata_cafe1.csv"
        self.dish_cafe2_file = "/dish_metadata_cafe2.csv"
        self.dish_ingredients_file = '/ingredients_metadata.csv'
        self.dish_id_col = ["dish_id"]