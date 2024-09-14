import matplotlib.pyplot as plt
import numpy as np
import json

models = ["relu_64_28_64", "relu_256_512_128", "sigmoid_64_128_64", "sigmoid_256_512_128"]
for name in models:
    with open("history_"+name+".json", "r") as f:
        loss_data = json.loads(f.read())
    plt.plot(range(len(loss_data['loss'])), loss_data['loss'], label=name)
    plt.legend()
plt.savefig('history_77_plot.png')
