import tensorflow as tf

print("TF :", tf.__version__)
print("Keras :", tf.keras.__version__)

model = tf.keras.models.load_model(
    "cnn_model_optimized_extra.keras",
    compile=False,
    safe_mode=False,
)

print("Model berhasil dimuat")