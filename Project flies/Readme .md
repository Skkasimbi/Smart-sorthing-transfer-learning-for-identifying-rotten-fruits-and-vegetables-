dataset/
├── train/
│   ├── fresh_apple/
│   ├── rotten_apple/
│   ├── fresh_tomato/
│   └── rotten_tomato/
├── val/
└── test/
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMAGE_SIZE + (3,),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False  # freeze base

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)
import matplotlib.pyplot as plt

# Accuracy plot
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.legend()
plt.title("Accuracy over epochs")
plt.show()
model.save('smart_sort_model.h5')

# Load later
# model = tf.keras.models.load_model('smart_sort_model.h5')
# streamlit_app.py
import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('smart_sort_model.h5')

def predict(image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    return preds

st.title("Fruit & Veggie Rotten Classifier")
img = st.file_uploader("Upload an image...", type=["jpg", "png"])

if img:
    image = Image.open(img)
    st.image(image, caption="Uploaded Image")
    prediction = predict(image)
    st.write("Prediction:", prediction)


