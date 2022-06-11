import argparse

import numpy as np
import json
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub


def prepare_image(image_path):
    """
    INPUT:
    image_path - string, path to image file

    OUTPUT:
    image - numpy array, processed image with shape (224, 224, 3)
    """
    image_size = 224
    image = Image.open(image_path)
    image = np.asarray(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    image = image.numpy()
    return image


def predict(image_path, model, top_k, category_names):
    """
    INPUT:
    image_path - string, path to image file
    model - object, model to make prediction
    top_k - integer, number of results to return

    OUTPUT:
    (returns the top_k most likely class labels along with the probabilities)
    probs - numpy array, probabilities
    classes - list, class labels
    """

    new_image = prepare_image(image_path)
    new_batch = np.expand_dims(new_image, axis=0)
    prediction = model.predict(new_batch)[0]

    ind = np.argsort(prediction)
    top_k_ind = ind[-top_k:]
    probs = prediction[top_k_ind]
    classes = top_k_ind + 1
    if category_names:
        with open(category_names, 'r') as f:
            class_names = json.load(f)
        classes = [class_names[str(n)] for n in classes]
    return probs, classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', help='Path to image file', type=str)
    parser.add_argument('model_path', help='Path to saved model', type=str)
    parser.add_argument('-k', '--top_k', help='Return the top K most likely classes', type=int, default=1)
    parser.add_argument('-c', '--category_names', help='Path to a JSON file mapping labels to flower names', type=str)

    args = parser.parse_args()

    print(args.img_path)
    print(args.top_k)

    model = tf.keras.models.load_model(args.model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    probs, classes = predict(args.img_path, model, args.top_k, args.category_names)

    for prob, cls in zip(list(probs)[::-1], list(classes)[::-1]):
        print(f'{cls}: {prob}')


if __name__ == '__main__':
    main()
