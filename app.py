from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import joblib
import tensorflow as tf
from flask_cors import CORS
import gdown


app = Flask(__name__)
CORS(app)


model_ids = {
    'inception': '1JQQkD-8jRQXI9Twp7BjJu_kxlmOqiljy',
    'densenet': '1OYIlqOvuQgD4OLKA0HAa5HCb_GovcWvI',
    'vgg16': '1bHoCvQIs7_jI12W50CyREDZHKnZwavnt'
}


class_labels = [
    "Tomato Bacterial spot", 
    "Tomato Early blight", 
    "Tomato Healthy",
    "Tomato Late blight", 
    "Tomato Leaf Mold", 
    "Tomato Mosaic virus", 
    "Tomato Septoria leaf spot",
    "Tomato Spider mites", 
    "Tomato Target Spot", 
    "Tomato Yellow Leaf Curl Virus"
]


models = {}
for name, model_id in model_ids.items():
    model_path = f'{name}_model.sav'
    gdown.download(f'https://drive.google.com/uc?id={model_id}', model_path, quiet=False)
    models[name] = joblib.load(model_path)


def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def get_model_prediction(model, image):
    prediction = model.predict(image)
    return np.argmax(prediction, axis=1)[0]


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    try:
        image = Image.open(image_file)
    except:
        return jsonify({'error': 'Invalid image format'}), 400

    preprocessed_image = preprocess_image(image, target_size=(224, 224))

    model_choice = request.args.get('model', 'all')

    if model_choice == 'all':
        predictions = {name: class_labels[get_model_prediction(model, preprocessed_image)] for name, model in models.items()}
        return jsonify(predictions)

    selected_model = models.get(model_choice)
    if selected_model:
        prediction = class_labels[get_model_prediction(selected_model, preprocessed_image)]
        return jsonify({f'{model_choice}_prediction': prediction})

    return jsonify({'error': f'Invalid model choice. Available options are: {", ".join(models.keys())}, or "all".'}), 400


@app.route('/status', methods=['GET'])
def status():
    return jsonify({"success": True, "code": 200})


if __name__ == '__main__':
    app.run(debug=True)