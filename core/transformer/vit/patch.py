import tensorflow as tf
from vit_config import VITConfig


class Patches(tf.keras.layers.Layer):
    def __init__(self, vit_config: VITConfig):
        super(Patches, self).__init__()
        self.vit_config = vit_config

    #     Override function to avoid error while saving model
    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "input_shape": self.vit_config.input_shape,
                "patch_size": self.vit_config.patch_size,
                "num_patches": self.vit_config.num_patches,
                "projection_dim": self.vit_config.projection_dim,
                "num_heads": self.vit_config.num_heads,
                "transformer_units": self.vit_config.transformer_units,
                "transformer_layers": self.vit_config.transformer_layers,
                "mlp_head_units": self.vit_config.mlp_head_units,
            }
        )
        return config

    def call(self, images):
        batch = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.vit_config.patch_size, self.vit_config.patch_size, 1],
            strides=[1, self.vit_config.patch_size, self.vit_config.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        # return patches
        return tf.reshape(patches, [batch, -1, patches.shape[-1]])
