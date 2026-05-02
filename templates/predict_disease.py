import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("disease_model.h5")

img = Image.open("test.jpg").resize((128,128))
img = np.array(img)/255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
print("Disease:", np.argmax(prediction))