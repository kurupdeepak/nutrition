import tensorflow as tf

from core.transformer.swin import SwinConfig, PatchExtract, PatchEmbedding, SwinTransformer, PatchMerging
from core.transformer.vit import VITConfig, Patches, PatchEncoder, mlp


class ModelCreator:

    @staticmethod
    def create_vit(vit_config: VITConfig):
        inputs = tf.keras.layers.Input(shape=vit_config.input_shape)
        # Data Augmentation Layers
        augmented = vit_config.data_augmentation_layers(inputs)
        # Create patches
        patches = Patches(vit_config)(inputs)
        # Encode patches
        encoded_patches = PatchEncoder(vit_config)(patches)

        # Create multiple layers of the Transformer block.
        for _ in range(vit_config.transformer_layers):
            # Layer normalization 1.
            x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = tf.keras.layers.MultiHeadAttention(
                num_heads=vit_config.num_heads, key_dim=vit_config.projection_dim, dropout=0.1
            )(x1, x1)
            # Skip connection 1.
            x2 = tf.keras.layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP
            x3 = mlp(x3, hidden_units=vit_config.transformer_units, dropout_rate=0.1)
            # Skip connection 2.
            encoded_patches = tf.keras.layers.Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = tf.keras.layers.Flatten()(representation)
        representation = tf.keras.layers.Dropout(0.3)(representation)
        # Add MLP.
        features = mlp(representation, hidden_units=vit_config.mlp_head_units, dropout_rate=0.3)

        logits = tf.keras.layers.Dense(vit_config.output_size)(features)
        # return Keras model.
        return tf.keras.Model(inputs=inputs, outputs=logits)

    @staticmethod
    def create_swin(swin_config: SwinConfig,
                    shift_size=1):
        input = tf.keras.layers.Input(swin_config.input_shape)
        # x = tf.keras.layers.RandomCrop(image_dimension, image_dimension)(input)
        # x = tf.keras.layers.RandomFlip("horizontal")(x)
        x = PatchExtract(swin_config.patch_size)(input)
        x = PatchEmbedding(swin_config.num_patch_x * swin_config.num_patch_y, swin_config.embed_dim)(x)
        x = SwinTransformer(swin_config)(x)
        swin_config_shift_1 = SwinConfig(dim=swin_config.embed_dim,
                                         num_patch=(swin_config.num_patch_x, swin_config.num_patch_y),
                                         num_heads=swin_config.num_heads,
                                         window_size=swin_config.window_size,
                                         shift_size=shift_size,
                                         num_mlp=swin_config.num_mlp,
                                         qkv_bias=swin_config.qkv_bias,
                                         dropout_rate=swin_config.dropout_rate)
        x = SwinTransformer(swin_config_shift_1)(x)
        x = PatchMerging((swin_config.num_patch_x, swin_config.num_patch_y), embed_dim=swin_config.embed_dim)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        output = tf.keras.layers.Dense(swin_config.output_size)(x)
        return tf.keras.Model(input, output)
