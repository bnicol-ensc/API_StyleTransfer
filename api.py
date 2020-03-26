from __future__ import absolute_import, division, print_function, unicode_literals
try:
  # %tensorflow_version only exists in Colab.
  import tensorflow.compat.v2 as tf
except Exception:
  pass
tf.enable_v2_behavior()
import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import time
import functools

from PIL import Image

import os

from flask import Flask, send_file, request, flash
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './img'
SAVE_FOLDER = './save'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SAVE_FOLDER'] = SAVE_FOLDER

content_path = tf.keras.utils.get_file('belfry.jpg','https://storage.googleapis.com/khanhlvg-public.appspot.com/arbitrary-style-transfer/belfry-2611573_1280.jpg')
style_path = tf.keras.utils.get_file('style23.jpg','https://storage.googleapis.com/khanhlvg-public.appspot.com/arbitrary-style-transfer/style23.jpg')

style_predict_path = tf.keras.utils.get_file('style_predict.tflite', 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/arbitrary_style_transfer/style_predict_quantized_256.tflite')
style_transform_path = tf.keras.utils.get_file('style_transform.tflite', 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/arbitrary_style_transfer/style_transfer_quantized_dynamic.tflite')

# Téléchargement de l'image et conversion de format
def load_img(path_to_img):
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  img = img[tf.newaxis, :]

  return img

# Pre-process style image input.
def preprocess_style_image(style_image):
  # Rediemmensionnement de l'image a 256px
  target_dim = 256
  shape = tf.cast(tf.shape(style_image)[1:-1], tf.float32)
  short_dim = min(shape)
  scale = target_dim / short_dim
  new_shape = tf.cast(shape * scale, tf.int32)
  style_image = tf.image.resize(style_image, new_shape)

  # Cut central
  style_image = tf.image.resize_with_crop_or_pad(style_image, target_dim, target_dim)

  return style_image

# Pre-process content image input.
def preprocess_content_image(content_image):
  # Cut central
  shape = tf.shape(content_image)[1:-1]
  short_dim = min(shape)
  content_image = tf.image.resize_with_crop_or_pad(content_image, short_dim, short_dim)

  return content_image

def imagesave(image, title):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  conv = tf.image.convert_image_dtype(image, dtype=tf.uint16)
  enc = tf.image.encode_png(conv, -1)
  tf.io.write_file(os.path.join(app.config['SAVE_FOLDER'], title),enc)
  

# Style prediction on preprocessed style image.
def run_style_predict(preprocessed_style_image):
  # Load the model.
  interpreter = tf.lite.Interpreter(model_path=style_predict_path)

  # Set model input.
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  interpreter.set_tensor(input_details[0]["index"], preprocessed_style_image)

  # Calculate style bottleneck.
  interpreter.invoke()
  style_bottleneck = interpreter.tensor(
      interpreter.get_output_details()[0]["index"]
      )()

  return style_bottleneck

def run_style_transform(style_bottleneck, preprocessed_content_image):
  # Load the model.
  interpreter = tf.lite.Interpreter(model_path=style_transform_path)

  # Set model input.
  input_details = interpreter.get_input_details()
  interpreter.resize_tensor_input(input_details[0]["index"],
                                  preprocessed_content_image.shape)
  interpreter.allocate_tensors()

  # Set model inputs.
  interpreter.set_tensor(input_details[0]["index"], preprocessed_content_image)
  interpreter.set_tensor(input_details[1]["index"], style_bottleneck)
  interpreter.invoke()

  # Transform content image.
  stylized_image = interpreter.tensor(
      interpreter.get_output_details()[0]["index"]
      )()

  return stylized_image


@app.route('/')
def hello():
    return {'hello': 'world'}

@app.route('/example')
def model():
    # Load the input images.
    print(content_path)
    content_image = load_img(content_path)
    style_image = load_img(style_path)

    # Preprocess the input images.
    preprocessed_content_image = preprocess_content_image(content_image)
    preprocessed_style_image = preprocess_style_image(style_image)

    style_bottleneck = run_style_predict(preprocessed_style_image)
    # Calculate style bottleneck of the content image.
    style_bottleneck_content = run_style_predict(
        preprocess_style_image(content_image)
        )

    # Define content blending ratio between [0..1].
    # 0.0: 0% style extracts from content image.
    # 1.0: 100% style extracted from content image.
    content_blending_ratio = 0.5

    # Blend the style bottleneck of style image and content image
    style_bottleneck_blended = content_blending_ratio * style_bottleneck_content \
                            + (1 - content_blending_ratio) * style_bottleneck

    # Stylize the content image using the style bottleneck.
    stylized_image_blended = run_style_transform(style_bottleneck_blended,
                                               preprocessed_content_image)

    # Save the output.
    imagesave(stylized_image_blended, 'example.png')

    return send_file(os.path.join(app.config['SAVE_FOLDER'], 'example.png'), mimetype='image/gif')

@app.route('/testInput', methods=['POST'])
def test():
    try:
        image = request.files['file']
        filename = secure_filename(image.filename)
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        path = (os.path.join(app.config['UPLOAD_FOLDER'], filename))
        content_image = load_img(path)
        style_image = load_img(style_path)

        preprocessed_content_image = preprocess_content_image(content_image)
        preprocessed_style_image = preprocess_style_image(style_image)

        style_bottleneck = run_style_predict(preprocessed_style_image)
        style_bottleneck_content = run_style_predict(
            preprocess_style_image(content_image)
            )

        content_blending_ratio = 0.5

        style_bottleneck_blended = content_blending_ratio * style_bottleneck_content \
                                + (1 - content_blending_ratio) * style_bottleneck

        stylized_image_blended = run_style_transform(style_bottleneck_blended,
                                                  preprocessed_content_image)

        imagesave(stylized_image_blended, filename)

        return send_file(os.path.join(app.config['SAVE_FOLDER'], filename), mimetype='image/gif')
    except Exception as err:
        return {'status':'not ok'}

if __name__ == '__main__':
    app.run(debug=True)