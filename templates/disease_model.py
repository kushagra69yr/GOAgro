import tensorflow as tf
from tensorflow.keras import layers, models
import os

# Dataset path (you must download PlantVillage dataset)
data_dir = "dataset/"

img_size = 128

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    image_size=(img_size, img_size),
    batch_size=32
)

model = models.Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(train_ds.class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds, epochs=5)

model.save("disease_model.h5")