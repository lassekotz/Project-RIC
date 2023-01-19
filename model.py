import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(4,)),
    tf.keras.layers.Dense(8)
])

model.compile(tf.keras.optimizers.RMSprop(0.001), loss='mse')
model.fit(np.zeros((10, 4)),
          np.ones((10, 8)))

model.summary()