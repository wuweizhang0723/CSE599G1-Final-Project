import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src import data, models


checkpoint_path = './output/tf/2_66_6_1_1024_7_512_5_2_0.0001_2_256/epoch=9-step=73440.ckpt'

attention_layers = 2
conv_layers = 6
conv_repeat = 1
filter_number = 512
h_layers = 2
hidden_size = 256
kernel_length = 7
kernel_number = 1024
kernel_size = 5
learning_rate = 0.0001
num_rel_pos_features = 66
pooling_size = 2


trainloader, valloader, testloader = data.load_data()
single_model = models.Transformer(
    kernel_number=kernel_number,
    kernel_length=kernel_length,
    filter_number=filter_number,
    kernel_size=kernel_size,
    pooling_size=pooling_size,
    conv_layers=conv_layers,
    conv_repeat=conv_repeat,
    attention_layers=attention_layers,
    hidden_size=hidden_size,
    dropout=0.2,
    h_layers=h_layers,
    pooling_type="avg",
    learning_rate=learning_rate,
    num_rel_pos_features=num_rel_pos_features,
)

trainer = pl.Trainer(
    devices=[2],
    accelerator="gpu",
    benchmark=False,
    profiler="simple",
)

# predictions = trainer.test(single_model, dataloaders=[valloader], ckpt_path=checkpoint_path)
# val_predictions = predictions[0]
# test_predictions = predictions[1]
# print(predictions)

# val_precision = val_predictions['precision']
# val_recall = val_predictions['recall']
# val_loss = val_predictions['test_loss']
# print(val_precision)
# print(val_recall)
# print(val_loss)

# test_precision = test_predictions['precision']
# test_recall = test_predictions['recall']
# test_loss = test_predictions['test_loss']
# print(test_precision)
# print(test_recall)
# print(test_loss)


predictions = trainer.predict(single_model, dataloaders=[valloader], ckpt_path=checkpoint_path)
predictions = torch.cat(predictions, dim=0).cpu().numpy()

all_pred = predictions[:, :164]
all_truth = predictions[:, 164:]

print(all_pred.shape)

pred = all_truth[:, 120]
truth = all_truth[:, 120]
fpr, tpr, thresholds = metrics.roc_curve(truth, pred)
print(fpr)
print(tpr)
score = metrics.roc_auc_score(truth, pred)
print('val roc auc score:', score)

plt.plot(fpr, tpr, "b-")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.show()
