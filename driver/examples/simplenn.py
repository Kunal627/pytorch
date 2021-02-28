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
from models.simplelinearnn import SimpleLinearNN

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]

t_c = torch.tensor(t_c).unsqueeze(1)
t_u = torch.tensor(t_u).unsqueeze(1)

t_u = t_u * .1

nsamples = t_u.shape[0]
nval = int(.3 * nsamples)
shfflidx = torch.randperm(nsamples)
trainidx = shfflidx[: -nval]
validx   = shfflidx[-nval :]
print(shfflidx, trainidx, validx)
trainX = t_u[trainidx]
trainY = t_c[trainidx]
valX   = t_u[validx]
valY   = t_c[validx]

print(t_c.shape)

model = SimpleLinearNN.build(1,13,1)
print(model)

optimizer = SGD(model.parameters(), lr=1e-4)

# pass complete set in both train and validation
training_loop(5000, optimizer, model, nn.MSELoss(), trainX, trainY, valX, valY)
#training_loop(10000, optimizer, model, nn.MSELoss(), t_u, t_c, t_u, t_c)
#
print('output', model(t_u))
print('answer', t_c)
print('hidden', model.hidden_linear.weight.grad)





#print(model)
#print([param.shape for param in model.parameters()])

#for name, param in model.named_parameters():
#    print(name, param.shape)

#tu = .1 * t_u

