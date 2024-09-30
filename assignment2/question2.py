import json
import tensorflow as tf
import time

train, validation = tf.keras.preprocessing.image_dataset_from_directory(
        directory="data/train/",
        label_mode="categorical",
        image_size=(220,220),
        color_mode="grayscale",
        validation_split=.1,
        subset="both",
        seed=1,
        batch_size=32,
        )
test = tf.keras.preprocessing.image_dataset_from_directory(
        directory="data/test/",
        label_mode="categorical",
        image_size=(220,220),
        color_mode="grayscale",
        )

model = tf.keras.models.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(.15),
    tf.keras.layers.Conv2D(32,(3,3),padding='same', activation="elu", input_shape=(220,220,1), kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32,(3,3),padding='same', activation="elu", input_shape=(220,220,1), kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Dropout(.2),

    tf.keras.layers.Conv2D(64,(3,3),padding='same', activation="elu", kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64,(3,3),padding='same', activation="elu", kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Dropout(.3),

    tf.keras.layers.Conv2D(128,(3,3),padding='same', activation="elu", kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128,(3,3),padding='same', activation="elu", kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Dropout(.4),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=15, activation="softmax")
    ])

learning_rate = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=500,
    alpha=0.1
)

optimizer = tf.keras.optimizers.SGD(lr=0.0001, decay=1e-5)

#optimizer = tf.keras.optimizers.RMSprop(lr=0.0001,decay=1e-5)

#optimizer = tf.keras.optimizers.Adam(lr=0.0001, decay=1e-5)

model.compile(
        loss = 'categorical_crossentropy',
        metrics = ["accuracy"],
        optimizer = optimizer
        )

results = model.evaluate(
        test,
        return_dict=True
        )

print(results)
start = time.time()
history = model.fit(
            train,
            validation_data=validation,
            epochs=200,
          )
history.history['training_time'] = time.time() - start
with open('question_2_arch1_SGD_history.json', 'w') as f:
    json.dump(history.history, f)
results = model.evaluate(
        test,
        return_dict=True
        )

print(results)
with open('question_2_arch1_SGD_results.json', 'w') as f:
    json.dump(results, f)
