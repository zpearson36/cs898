import matplotlib.pyplot as plt
import numpy as np
import json

models = ["RNN", "LSTM", "GRU"]
figure, axis = plt.subplots(2,2)
axes = [[0,0],[0,1],[1,0],[1,1]]
i = 0
xlabel = "Epoch"
ylabel = "Accuracy"
for name, ax in zip(models,axes):
    with open(name+"_history.json", "r") as f:
        loss_data = json.loads(f.read())
#    axis[ax[0]][ax[1]].plot(np.linspace(0,int(loss_data['training_time']), len(loss_data['loss'])), loss_data['loss'], label="Training Loss")
#    axis[ax[0]][ax[1]].plot(np.linspace(0,int(loss_data['training_time']), len(loss_data['val_loss'])), loss_data['loss'], label="Validation Loss")
    axis[ax[0]][ax[1]].plot(range(len(loss_data['accuracy'])), loss_data['accuracy'], label="Training Acccuracy")
    axis[ax[0]][ax[1]].plot(range(len(loss_data['val_accuracy'])), loss_data['val_accuracy'], label="Validation Acccuracy")
    axis[ax[0]][ax[1]].set_title(name)
    axis[ax[0]][ax[1]].set_xlabel(xlabel + " (s)")
    axis[ax[0]][ax[1]].set_ylabel(ylabel)
    axis[ax[0]][ax[1]].legend(["Training", "Validation"])
figure.suptitle("Training Accuracy vs Training Epochs")    
plt.tight_layout()
plt.savefig(f'{ylabel}_per_{xlabel}_plot.png')
