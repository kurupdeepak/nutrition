import tensorflow as tf
from vit_config import VITConfig


class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, vit_config: VITConfig):
        super(PatchEncoder, self).__init__()
        self.vit_config = vit_config
        self.projection = tf.keras.layers.Dense(units=vit_config.projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=vit_config.num_patches, output_dim=vit_config.projection_dim
        )

    # Override function to avoid error while saving model
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

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
