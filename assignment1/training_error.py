import matplotlib.pyplot as plt
import numpy as np
import json

models = ["relu_64_28_64", "relu_256_512_128", "sigmoid_64_128_64", "sigmoid_256_512_128"]
for name in models:
    with open("history_"+name+".json", "r") as f:
        loss_data = json.loads(f.read())
    plt.plot(list(np.arange(0, loss_data["training_time"],loss_data["training_time"]/len(loss_data['val_mean_squared_error']))), [x + y for x,y in zip(loss_data['mean_squared_error'],loss_data['val_mean_squared_error'])], label=name)
    plt.legend()
plt.savefig('tot_error_77_plot.png')
