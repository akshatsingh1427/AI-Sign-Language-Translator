import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

X = []
y = []

labels = os.listdir("dataset2")

for idx, label in enumerate(labels):
    for file in os.listdir(f"dataset2/{label}"):
        data = np.load(f"dataset2/{label}/{file}")
        X.append(data.flatten())
        y.append(idx)

X = np.array(X)
y = np.array(y)

model = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(63,)),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(labels), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X, y, epochs=20)

model.save("landmark_model.h5")
np.save("labels.npy", labels)
