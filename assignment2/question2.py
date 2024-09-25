import tensorflow as tf

train, test =  tf.keras.preprocessing.image_dataset_from_directory(
        directory="data/",
        label_mode="categorical",
        image_size=(220,220),
        color_mode="grayscale",
        crop_to_aspect_ratio=True,
        validation_split=.2,
        subset="both",
        shuffle=False,
        )
#data_train, labels_train = train.element_spec
#data_val, labels_val =  val.element_spec
#print(list(train))
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,5,(1,1),'same', activation="relu", input_shape=(220,220,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(64,3,(1,1),'same', activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512,activation="relu"),
    tf.keras.layers.Dense(units=15, activation="softmax")
    ])

learning_rate = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.1,
    decay_steps=3,
    alpha=0.001
)

optimizer = tf.keras.optimizers.SGD(
    learning_rate=learning_rate,
    momentum=0.9
)

model.compile(
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics = [tf.keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None), "accuracy"],
    optimizer = optimizer
        )
model.summary()

results = model.evaluate(
        test,
        batch_size=32,
        return_dict=True
        )

print(results)

history = model.fit(
            train,
            batch_size=32,
            epochs=10,
          )
print(history.history)
results = model.evaluate(
        test,
        batch_size=32,
        return_dict=True
        )

print(results)
