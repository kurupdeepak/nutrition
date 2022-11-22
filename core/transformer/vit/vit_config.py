class VITConfig:
    def __init__(self,
                 image_size=None,
                 patch_size=None,
                 projection_dim=None,
                 num_heads=None,
                 transformer_layers=None,
                 mlp_heads=None,
                 output_shape=None):
        self.image_size = image_size
        self.patch_size = patch_size
        self.input_shape = (image_size, image_size, 3)  # input image shape
        self.num_patches = (image_size // patch_size) ** 2
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        # Size of the transformer layers
        self.transformer_units = [
            projection_dim * 2,
            projection_dim,
        ]
        self.transformer_layers = transformer_layers
        self.mlp_head_units = mlp_heads  # Size of the dense layers
        self.output_size = output_shape

    def data_augmentation_layers(self, inputs):
        pass
