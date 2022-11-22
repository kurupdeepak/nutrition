from core.vit.vit_config import *
import tensorflow as tf


class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, no_of_patches, proj_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = no_of_patches
        self.projection = tf.keras.layers.Dense(units=proj_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=no_of_patches, output_dim=proj_dim
        )

    # Override function to avoid error while saving model
    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "input_shape": input_shape,
                "patch_size": patch_size,
                "num_patches": num_patches,
                "projection_dim": projection_dim,
                "num_heads": num_heads,
                "transformer_units": transformer_units,
                "transformer_layers": transformer_layers,
                "mlp_head_units": mlp_head_units,
            }
        )
        return config

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
