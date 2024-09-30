import json
import tensorflow as tf
import time

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,5,(3,3),'same', activation="relu", input_shape=(32,32,3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(64,3,(3,3),'same', activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512, activation="relu"),
    tf.keras.layers.Dense(units=10, activation="softmax")
    ])

(xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.cifar10.load_data()
ytrain=tf.keras.utils.to_categorical(ytrain)
ytest=tf.keras.utils.to_categorical(ytest)
print(xtrain.shape)
print(xtest.shape)
print(ytest.shape)
print(ytrain.shape)
print(xtrain.shape[0]*.8*100)
learning_rate = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.1,
    decay_steps=xtrain.shape[0]*.8*100,
    alpha=0.01
)

optimizer = tf.keras.optimizers.SGD(
    learning_rate=learning_rate,
    momentum=.1
)

model.compile(
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics = [tf.keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None), "accuracy"],
    optimizer = optimizer
        )
model.summary()
results = model.evaluate(
        x=xtest,
        y=ytest,
        batch_size=32,
        return_dict=True
        )

print(results)
start = time.time()
history = model.fit(
            x=xtrain,
            y=ytrain,
            batch_size=32,
            epochs=100,
            validation_split=.2
          )
print(history.history)
history.history["training_time"] = time.time() - start
with open("history.json", "w") as f:
    json.dump(history.history, f)
results = model.evaluate(
        x=xtest,
        y=ytest,
        batch_size=32,
        return_dict=True
        )

print(results)
with open("results.json", "w") as f:
    json.dump(history.history, f)
