import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,5,(1,1),'same', activation="relu", input_shape=(32,32,3)),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(64,3,(1,1),'same', activation="relu"),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512,activation="relu"),
    tf.keras.layers.Dense(units=10, activation="softmax")
    ])

learning_rate = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.1,
    decay_steps=1000,
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
(xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.cifar10.load_data()
ytrain=tf.keras.utils.to_categorical(ytrain)
ytest=tf.keras.utils.to_categorical(ytest)
print(xtrain.shape)
print(xtest.shape)
print(ytest.shape)
print(ytrain.shape)
results = model.evaluate(
        x=xtest,
        y=ytest,
        batch_size=32,
        return_dict=True
        )

print(results)

history = model.fit(
            x=xtrain,
            y=ytrain,
            batch_size=32,
            epochs=10,
            validation_split=.2
          )
print(history.history)

results = model.evaluate(
        x=xtest,
        y=ytest,
        batch_size=32,
        return_dict=True
        )

print(results)
