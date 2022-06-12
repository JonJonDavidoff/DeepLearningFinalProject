from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
from PIL import Image
import io
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
import numpy as np
from keras.applications.vgg16 import preprocess_input
import pdb
import tensorflow as tf


num_classes = 6
# create model
improved_model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150,3)),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes, activation='softmax')
])

# compile model and load weights

improved_model.compile()
improved_model.load_weights('static/model_classifier.h5')

# the function loads images for the model's specific format
def load_iamge_for_model(img_bytes):
	target_size = (150,150)
	img = Image.open(io.BytesIO(img_bytes)) # tansfor to image object from bytes
	img = img.convert('RGB') # convert to rgb format
	img = img.resize(target_size, Image.NEAREST)  # resize image 
	img = img_to_array(img) 
	img = np.expand_dims(img, axis=0)
	img = preprocess_input(img)
	img /= 255. #rescale to 0-1
	return img

# the predict function gets an image and returns the model's predction
def predict(img):
	results = improved_model.predict(img)
	final_result = np.argmax(results[0])
	print("The predction is: ", final_result)
	return final_result