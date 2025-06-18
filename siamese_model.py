from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class L2Norm(Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

@tf.keras.utils.register_keras_serializable()
class EuclideanDistance(Layer):
    def call(self, inputs):
        x, y = inputs
        return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))

@tf.keras.utils.register_keras_serializable()
def contrastive_loss(margin=1.0):
    def loss(y_true, y_pred):
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
    return loss

def build_embedder(input_shape, embed_dim=128, use_l2norm=True):
    input = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(embed_dim)(x)

    if use_l2norm:
        x = L2Norm(name="l2_norm")(x)

    return models.Model(input, x, name="embedder")

def build_siamese(input_shape, embed_dim=128, use_l2norm=True, base_model=None):
    if base_model is not None:
        embedder = base_model
    else:
        embedder = build_embedder(input_shape, embed_dim=embed_dim, use_l2norm=use_l2norm)

    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)

    vec_a = embedder(input_a)
    vec_b = embedder(input_b)

    distance = CosineDistance(name="cosine_distance")([vec_a, vec_b])

    siamese_model = models.Model([input_a, input_b], distance)
    return siamese_model, embedder

@tf.keras.utils.register_keras_serializable()
class CosineDistance(Layer):
    def call(self, inputs):
        x, y = inputs
        x = tf.math.l2_normalize(x, axis=1)
        y = tf.math.l2_normalize(y, axis=1)
        cosine_similarity = tf.reduce_sum(x * y, axis=1, keepdims=True)
        return 1.0 - cosine_similarity  # distancia coseno