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

path = '../data-unversioned/p1ch7/'

# transform image to tensor
#cifar10 = datasets.CIFAR10(path, train=True, download=True, transform=transforms.ToTensor())
cifar10_val = datasets.CIFAR10(path, train=False, download=True, transform=transforms.ToTensor())

#class_names = ['airplane', 'automobile', 'bird', 'cat' , 'deer', 'dog' , 'frog' , 'horse' , 'ship',  'truck']

#imgs = torch.stack([img_t for img_t, _ in cifar10], dim=3)
#print(imgs.shape)

# calculate mean and std per channel for the complete train dataset
#mean = imgs.view(3, -1).mean(dim=1)
#stddev = imgs.view(3, -1).std(dim=1)

#print(mean, stddev)

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))])
transformed_cifar10 = datasets.CIFAR10(path, train=True, download=False, 
                                       transform=trans )


# get only aeroplanes and birds from train and val sets
label_map = {0: 0, 2: 1}
class_names = ['airplane', 'bird']
cifar2 = [(img, label_map[label]) for img, label in transformed_cifar10 if label in [0, 2]]
cifar2_val = [(img, label_map[label]) for img, label in cifar10_val if label in [0, 2]]

train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64,shuffle=True)
val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64,shuffle=False)

# 32 * 32 * 3 = 3072 features , 512 is random choice for hidden layer and 2 classes - aeroplanes and birds

model = nn.Sequential(
    nn.Linear(3072, 1024),
    nn.Tanh(),
    nn.Linear(1024, 512),
    nn.Tanh(),
    nn.Linear(512, 128),
    nn.Tanh(),
    nn.Linear(128, 2),
    nn.LogSoftmax(dim=1))
    
loss_fn = nn.NLLLoss()

learning_rate = 1e-2
optimizer = SGD(model.parameters(), lr=learning_rate)

n_epochs = 100
# train for 100 epochs
for epoch in range(n_epochs):
    for imgs, labels in train_loader:
        batch_size = imgs.shape[0]
        out = model(imgs.view(batch_size, -1))
        loss = loss_fn(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))

correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in val_loader:
        batch_size = imgs.shape[0]
        outputs = model(imgs.view(batch_size, -1))
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())

print("Accuracy: %f", correct / total)

#img, label = cifar2[0]
#out = model(img.view(-1).unsqueeze(0))
#print(loss(out, torch.tensor([label])))


#img, _ = cifar2[0]
#plt.imshow(img.permute(1, 2, 0))
#plt.show()

#img_batch = img.view(-1).unsqueeze(0)
#print(img_batch.shape)

#out = model(img_batch)
#print(out)
#img_t, _ = transformed_cifar10[99]
#plt.imshow(img_t.permute(1, 2, 0))
#plt.show()

#img, label = cifar10[99]
#print(img, label, class_names[label])
#
#
#print(img.shape)
## permute to change from C X H X W to H X W X C
#plt.imshow(img.permute(1, 2, 0))
#plt.show()
