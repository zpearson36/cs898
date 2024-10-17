import numpy as np
import os
import tensorflow as tf
from PIL import Image

dir_list = os.listdir("data/train")

train_dirty = []
train_clean = []
for im in dir_list:
    dirty = np.asarray(Image.open("data/train/"+im), dtype=np.float32)
    clean = np.asarray(Image.open("data/train_cleaned/"+im), dtype=np.float32)
    xoff = 540 - dirty.shape[0]
    yoff = 540 - dirty.shape[1]
    dirty = np.pad(dirty, [(0,xoff), (0,yoff)], mode="constant")
    clean = np.pad(clean, [(0,xoff), (0,yoff)], mode="constant")
    train_dirty.append(dirty.reshape(540,540,1))
    train_clean.append(clean.reshape(540,540,1))

train_dirty = np.array(train_dirty)
train_dirty.shape

train_clean = np.array(train_clean)
train_clean.shape

dir_list = os.listdir("data/test")
test = []
for im in dir_list:
    test_dirty = np.asarray(Image.open("data/test/"+im))
    xoff = 540 - test_dirty.shape[0]
    yoff = 540 - test_dirty.shape[1]
    test_dirty = np.pad(test_dirty, [(0,xoff), (0,yoff)], mode="constant")
    test.append(test_dirty.reshape(540,540,1))

test = np.array(test)
test.shape

encoder_decoder = tf.keras.models.Sequential([
    # encoder
    tf.keras.layers.Input(shape=(540,540,1)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
    tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),

    # decoder
    tf.keras.layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
    tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
    tf.keras.layers.Conv2D(1, kernel_size=(3, 3), activation='softplus', padding='same'),
    ])

def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))

optimizer = tf.keras.optimizers.Adam()
encoder_decoder.compile(
        loss = ssim_loss,
        metrics = ["accuracy"],
        optimizer = optimizer
        )
encoder_decoder.output_shape
import time
start = time.time()
history = encoder_decoder.fit(train_dirty, train_clean, batch_size=8, epochs=10,validation_split=.1)
history.history["training_time"] = time.time() - start

import json
with open("denoise_10_epochs_history.json", "w") as f:
    json.dump(history.history, f)

t = encoder_decoder(test[0].reshape(-1, 540, 540, 1))
t = np.array(t).reshape(540,540)

from matplotlib import pyplot as plt
plt.imshow(t, interpolation='nearest')
plt.show()
