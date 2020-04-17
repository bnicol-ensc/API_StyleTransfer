from styleTansfer import *

from PIL import Image

import os

from flask import Flask, send_file, request, flash
from werkzeug.utils import secure_filename
from markupsafe import escape

from flask_cors import CORS

UPLOAD_FOLDER = './upload'
SAVE_FOLDER = './save'
STYLE_FOLDER = './styles'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SAVE_FOLDER'] = SAVE_FOLDER
app.config['STYLE_FOLDER'] = STYLE_FOLDER


def imagesave(image, title):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    conv = tf.image.convert_image_dtype(image, dtype=tf.uint16)
    enc = tf.image.encode_png(conv, -1)
    tf.io.write_file(os.path.join(app.config['SAVE_FOLDER'], title), enc)


@app.route('/')
def hello():
    return {'AppName': 'Style Transfer API',
            'Author': 'Benjamin Nicol',
            'Version': 'v0.1',
            'Documentation': 'https://github.com/bnicol-ensc/API_StyleTransfer'}


@app.route('/example')
def example():
    # Load the input images.
    path = (os.path.join(app.config['UPLOAD_FOLDER'], 'belfry.jpg'))
    content_image = load_img(path)

    pathStyle = (os.path.join(app.config['STYLE_FOLDER'], 'udnie.jpg'))
    style_image = load_img(pathStyle)

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


@app.route('/model/<modelName>', methods=['POST'])
def model(modelName):
    try:
        image = request.files['file']
        #image = request.form['image']
        filename = secure_filename(image.filename)
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        path = (os.path.join(app.config['UPLOAD_FOLDER'], filename))
        content_image = load_img(path)

        styles = {
            'sphere': 'escher_sphere.jpg',
            'scream': 'scream.jpg',
            'udnie': 'udnie.jpg'
        }
        chosenStyle = styles.get(modelName)

        pathStyle = (os.path.join(app.config['STYLE_FOLDER'], chosenStyle))
        style_image = load_img(pathStyle)

        preprocessed_content_image = preprocess_content_image(content_image)
        preprocessed_style_image = preprocess_style_image(style_image)

        style_bottleneck = run_style_predict(preprocessed_style_image)
        style_bottleneck_content = run_style_predict(
            preprocess_style_image(content_image)
        )

        content_blending_ratio = 0.20

        style_bottleneck_blended = content_blending_ratio * style_bottleneck_content \
            + (1 - content_blending_ratio) * style_bottleneck

        stylized_image_blended = run_style_transform(style_bottleneck_blended,
                                                     preprocessed_content_image)

        imagesave(stylized_image_blended, filename)
        print("OKOKOK")
        return send_file(os.path.join(app.config['SAVE_FOLDER'], filename), mimetype='image/gif')
    except Exception as err:
        return {'status': 'not ok'}


if __name__ == '__main__':
    app.run(debug=True)
