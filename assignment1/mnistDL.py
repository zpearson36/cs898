import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

models      = ["relu_64_28_64", "relu_256_512_128", "sigmoid_64_128_64", "sigmoid_256_512_128"]
activations = ["relu", "relu", "sigmoid", "sigmoid"]
layers      = [(64,128,64), (256,512,128), (64,128,64), (256,512,128)]
for name, activation, layer in zip(models, activations, layers):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=layer[0],   activation=activation,input_shape=(784,)),
        tf.keras.layers.Dense(units=layer[1],  activation=activation),
        tf.keras.layers.Dense(units=layer[2],   activation=activation),
        tf.keras.layers.Dense(units=10,   activation='softmax'),
    ])
    model.compile(
            loss = tf.keras.losses.CategoricalCrossentropy(),
            metrics = [tf.keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None)]
            )
    
    print("="*20)
    print(tf.keras.layers.Flatten()(x_test)[0].shape)
    print(tf.keras.layers.Flatten()(x_test).shape)
    print(x_test.shape)
    print(x_test[0].shape)
    print(tf.keras.layers.Flatten()(x_test[0]).shape)
    print(y_test[0])
    print("="*20)
    tmp = []
    for y in y_test:
        tmp.append([0 for _ in range(10)])
        tmp[-1][y] = 1
    
    prediction = model.predict(x=tf.keras.layers.Flatten()(x_test))
    for i in range(10):
        np.argmax(prediction[i])
        print(f"guess: {np.argmax(prediction[i])}, expected: {y_test[i]}")
    results = model.evaluate(
            x=tf.keras.layers.Flatten()(x_test),
            y=np.array(tmp),
            batch_size=64,
            return_dict=True
            )
    print(results)
    
    tmp = []
    for y in y_train:
        tmp.append([0 for _ in range(10)])
        tmp[-1][y] = 1
    start = time.time()    
    model.summary()
    history = model.fit(
            x=tf.keras.layers.Flatten()(x_train),
            y=np.array(tmp),
            batch_size=64,
            epochs=500,
            validation_split=.2
            )
    history.history["training_time"] = time.time()-start
    print(history.history)
    with open("history_"+name+".json", "w") as f:
        json.dump(history.history, f)
    
    tmp = []
    for y in y_test:
        tmp.append([0 for _ in range(10)])
        tmp[-1][y] = 1
    
    results = model.evaluate(
            x=tf.keras.layers.Flatten()(x_test),
            y=np.array(tmp),
            batch_size=64,
            return_dict=True
            )
    
    print(results)
    with open("results_"+name+".json", "w") as f:
        json.dump(results, f)
    prediction = model.predict(x=tf.keras.layers.Flatten()(x_test))
    for i in range(10):
        np.argmax(prediction[i])
        print(f"guess: {np.argmax(prediction[i])}, expected: {y_test[i]}")
    model.save(name+".keras")
    plt.plot(range(len(history.history['loss'])), history.history['loss'], label=name)
    plt.legend()
plt.savefig('history_plot.png')
