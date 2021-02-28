from pathlib import Path # if you haven't already done so
import sys, os
file = Path(__file__).resolve()
parent, root, child = file.parent, file.parents[1], file.parents[2]
sys.path.append(str(root))
sys.path.append(str(child))


from torchvision import datasets
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
from torch.optim import SGD
import torch.nn as nn
from models.covnet import Covnet
from common.helper import trainloop

path = '../data-unversioned/p1ch7/'

# transform image to tensor
cifar10 = datasets.CIFAR10(path, train=True, download=True, transform=transforms.ToTensor())
cifar10_val = datasets.CIFAR10(path, train=False, download=True, transform=transforms.ToTensor())

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))])
transformed_cifar10 = datasets.CIFAR10(path, train=True, download=False, transform=trans )


# get only aeroplanes and birds from train and val sets
label_map = {0: 0, 2: 1}
class_names = ['airplane', 'bird']
cifar2 = [(img, label_map[label]) for img, label in transformed_cifar10 if label in [0, 2]]
cifar2_val = [(img, label_map[label]) for img, label in cifar10_val if label in [0, 2]]

train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64,shuffle=True)
val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64,shuffle=False)

model = Covnet()
loss_fn = nn.CrossEntropyLoss()

learning_rate = 1e-2
optimizer = SGD(model.parameters(), lr=learning_rate)


trainloop(100, optimizer=optimizer, model=model, loss_fn=loss_fn, train_loader=train_loader)