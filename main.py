import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.notebook import tqdm
import os
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from torch.utils.data import Subset
from torch.utils.data import DataLoader, RandomSampler
import LoadData.LoadDataset as LD
from model.wideresnet import WideResnet
import torch.optim
from algorithm import algorithm

if torch.cuda.is_available():
    train_on_gpu = True
else:
    train_on_gpu = False
device = torch.device("cuda")
print(torch.__version__)


label_dataset = LD.LoadLabeledData(
    "Label資料夾的路徑",
    color_mode="RGB"
)

unlabeled_dataset = LD.LoadUnlabeledData(
    "Unlabel資料及的路徑",
    color_mode = "RGB",
    mode='FixMatch'
)

test_dataset = LD.LoadTestData(
    "Test資料集的路徑"
)

def get_cos_schedule_with_warmup(optimizer, warmup_steps, training_steps, num_cycles=7./16., last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        no_progress = float(current_step - warmup_steps) / float(max(1, training_steps - warmup_steps))

        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

no_decay = ['bias', 'bn']
grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(
    nd in n for nd in no_decay)], 'weight_decay': 0.0005},
    {'params': [p for n, p in model.named_parameters() if any(
    nd in n for nd in no_decay)], 'weight_decay': 0.0}

]

def accuracy(output, target):
    _, predicted = torch.max(output, dim=1)
    total = target.shape[0]
    correct = int((predicted == target).sum())
    return correct/total

model = WideResnet(n_classes=3, proj=False) #Proj = True when using CoMatch

#optimizer = optim.SGD(grouped_parameters, lr=3e-2, momentum=0.9)
optimizer = optim.AdamW(model.parameters(), lr=2e-3)

optimizer.param_groups[0]['lr'] = 2e-3

scheduler = get_cos_schedule_with_warmup(optimizer, 10, 250)

loss_func = nn.CrossEntropyLoss()

mode = 'fixmatch'

target_epoch = 1

label_loader = DataLoader(label_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
unlabel_loader = DataLoader(unlabel_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)

for epoch in range(1, target_epoch+1):
    if mode == 'fixmatch':
        train_acc, labeled_loss, unlabeled_loss, pseudolabel_percentage= algorithm.train_fixmatch(
            model=model,
            loss_label=loss_func,
            metric=accuracy,
            optimizer=optimizer,
            label_loader=label_loader,
            unlabel_loader=unlabel_loader,
            threshold=0.95,
            train_on_gpu=True)