import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv3D(32,5,(1,1,1),0, activation="relu"),
    tf.keras.layers.MaxPool3D((2,2,1)),
    tf.keras.layers.Conv3D(64,3,(1,1,1),0  activation="Relu"),
    tf.keras.layers.MaxPool3d((2,2,1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512,activation="relu"),
    tf.keras.layers.Dense(units=10, activation="softmax")
    ])
