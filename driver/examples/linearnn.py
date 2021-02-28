from pathlib import Path # if you haven't already done so
import sys, os
file = Path(__file__).resolve()
parent, root, child = file.parent, file.parents[1], file.parents[2]
sys.path.append(str(root))
sys.path.append(str(child))


import torch
import torch.nn as nn
from torch.optim import SGD
from common.helper import training_loop

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]

t_c = torch.tensor(t_c).unsqueeze(1)
t_u = torch.tensor(t_u).unsqueeze(1)
t_u = t_u  * .1

nsamples = t_u.shape[0]
nval = int(0 * nsamples)

shfflidx = torch.randperm(nsamples)

trainidx = shfflidx[: -nval]
validx   = shfflidx[-nval :]

print(shfflidx, trainidx, validx)

trainX = t_u[trainidx]
trainY = t_c[trainidx]
valX   = t_u[validx]
valY   = t_c[validx]
print(trainX, trainY)
model = nn.Linear(1,1)
print(list(model.parameters()))
optimizer = SGD(model.parameters(), lr=0.01)

# pass complete set in both train and validation
training_loop(3000, optimizer, model, nn.MSELoss(), t_u, t_c, t_u, t_c)
print()
print(model.weight)
print(model.bias)