import tensorflow as tf

from utils import preprocess_image


def predict(image_path, save_model):
  image = preprocess_image(tf.io.read_file(image_path))
  model = tf.keras.models.load_model(save_model)
  return model(image)
