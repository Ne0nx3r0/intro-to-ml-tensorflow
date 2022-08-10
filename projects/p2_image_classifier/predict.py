import argparse

# Hide warnings, lot of chatter from Tensorflow about CPU usage
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Handle arguments
parser = argparse.ArgumentParser(description='ML-based flower type classifier for images')

# Required arguments
parser.add_argument("image_path", help="Path to the image to classify")
parser.add_argument("model", help="Model to use")

# Optional arguments
parser.add_argument("--top_k", help="Return top N most likely flower types", default="3")
parser.add_argument("--category_names", help="Path to a JSON file mapping labels to flower names", default="./label_map.json")

args = parser.parse_args()

# Script imports
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json
from PIL import Image

# Load the model
model = tf.keras.models.load_model(
    args.model,
    custom_objects={'KerasLayer':hub.KerasLayer}
)

# Image processor
image_size = 224

def process_image(imageIn):
    imageResized = tf.image.resize(imageIn,(image_size,image_size))
    return imageResized.numpy() / 255

# Prediction function
def predict(image_path, model, top_k):
    image = Image.open(image_path)
    imageArr = np.asarray(image)
    processed_image = process_image(imageArr)

    predictions = model.predict(np.expand_dims(processed_image,axis=0))
    
    topKResult = tf.math.top_k(
        predictions, k=top_k, sorted=True, name=None
    )
    
    return topKResult.indices.numpy()[0],topKResult.values.numpy()[0]

# Get results
classes,probabilities = predict(args.image_path, model, int(args.top_k)) 

# Load label maps
with open(args.category_names, "r") as file:
    classNamesRaw = json.load(file)

classNames = dict()

for label in classNamesRaw:
    zeroIndex = int(label) - 1
    classNames[zeroIndex] = classNamesRaw[label]

index = -1
for c in classes:
    index = index + 1

    flower = classNames[c]
    percent = int(probabilities[index] * 10000) / 100

    print(
        "{0} - {1}%".format(
            flower,
            percent
        )
    )