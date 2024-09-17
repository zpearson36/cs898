import matplotlib.pyplot as plt
import numpy as np
import json

models = ["relu_64_28_64", "relu_256_512_128", "sigmoid_64_128_64", "sigmoid_256_512_128"]
figure, axis = plt.subplots(2,2)
axes = [[0,0],[0,1],[1,0],[1,1]]
i = 0
xlabel = "Training Time"
ylabel = "Error"
for name, ax in zip(models,axes):
    with open("history_"+name+".json", "r") as f:
        loss_data = json.loads(f.read())
    axis[ax[0]][ax[1]].plot(np.linspace(0,int(loss_data['training_time']), len(loss_data['mean_squared_error'])), loss_data['mean_squared_error'], label="Training Error")
    axis[ax[0]][ax[1]].plot(np.linspace(0,int(loss_data['training_time']), len(loss_data['val_mean_squared_error'])), loss_data['val_mean_squared_error'], label="Validation Error")
    #axis[ax[0]][ax[1]].plot(range(len(loss_data['val_loss'])), loss_data['val_loss'], label="Validation Loss")
    axis[ax[0]][ax[1]].set_title(name)
    axis[ax[0]][ax[1]].set_xlabel(xlabel + " (s)")
    axis[ax[0]][ax[1]].set_ylabel(ylabel)
figure.suptitle("Training Error vs Training Time")    
plt.tight_layout()
plt.legend()
plt.savefig(f'{ylabel}_per_{xlabel}_plot_TaV.png')
