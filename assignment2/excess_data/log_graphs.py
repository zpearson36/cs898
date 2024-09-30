import matplotlib.pyplot as plt
import numpy as np
import json

models = ["question1"]
figure, axis = plt.subplots(1)
axes = [0]#[[0,0],[0,1],[1,0],[1,1]]
i = 0
xlabel = "Training Time"
ylabel = "Loss"
for optimizer, ax in zip(models,axes):
#    axis[ax].set_ylim([0,1])
    with open("history.json", "r") as f:
        loss_data = json.loads(f.read())
    axis.plot(np.linspace(0,int(loss_data['training_time']), len(loss_data['loss'])), loss_data['loss'], label="Training Loss")
    axis.plot(np.linspace(0,int(loss_data['training_time']), len(loss_data['val_loss'])), loss_data['val_loss'], label="Validation Loss")
    #axis.plot(range(len(loss_data['loss'])), loss_data['loss'], label="Training loss")
    #axis.plot(range(len(loss_data['val_loss'])), loss_data['val_loss'], label="Validation loss")
    #axis[ax[0]][ax[1]].plot(range(len(loss_data['val_loss'])), loss_data['val_loss'], label="Validation Loss")
    axis.set_title(optimizer)
    axis.set_xlabel(xlabel + " (s)")
    axis.set_ylabel(ylabel)
figure.suptitle("Training loss vs Training Time")    
plt.tight_layout()
#plt.legend()
plt.savefig(f'{ylabel}_per_{xlabel}_plot.png')

