import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
from keras import preprocessing
from keras.models import load_model
from keras.utils import img_to_array
import os
import h5py


st.title("The tomato disease classification")

def main():
    file_upload = st.file_uploader("Choose image", type=['jpeg', 'jpg'])
    if file_upload is not None:
        image = Image.open(file_upload)
        figure = plt.figure()
        plt.imshow(image)  # type: ignore
        plt.axis("off")
        result = predict_class(image)
        st.write(result)
        st.pyplot(figure)


def predict_class(image):
    class_mode = tf.keras.models.load_model('Tomato disease1.hdf5')
    shape = ((180, 180, 3))
    model = tf.keras.Sequential(
        [hub.KerasLayer(class_mode, input_shape=shape)])  # type: ignore
    test_image = image.resize((180, 180))
    test_img = img_to_array(test_image)
    test_image = np.array(test_image)/255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names = ['Tomato_Bacterial_spot',
                   'Tomato_Early_blight',
                   'Tomato_Late_blight',
                   'Tomato_Leaf_Mold',
                   'Tomato_Septoria_leaf_spot',
                   'Tomato_Spider_mites_Two_spotted_spider_mite',
                   'Tomato__Target_Spot',
                   'Tomato__Tomato_YellowLeaf__Curl_Virus',
                   'Tomato__Tomato_mosaic_virus',
                   'Tomato_healthy']
    prediction = model.predict(test_image)
    score = tf.nn.softmax(prediction[0])
    img_class = class_names[np.argmax(score)]
    result = "The image uploaded is :{} ".format(img_class)
    return result


if __name__ == "__main__":
    main()
