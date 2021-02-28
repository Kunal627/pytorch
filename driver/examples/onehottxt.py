

import torch

with open('.\data\exdata\prideandprejudice.txt') as f:
    data = f.read()

lines = data.split('\n')

line1 = '"Impossible, Mr. Bennet, impossible, when I am not acquainted with him him'
print(len(line1))

onehot_c = torch.zeros(len(line1), 128)

# convert the string to one hot encoding base on ASCII value of character
for i,c in enumerate(line1.lower().strip()):
    idx = ord(c) if ord(c) < 128 else 0
#    print(idx)
    onehot_c[i][idx] = 1

#print(onehot_c[2])


# convert the string to one hot encoding base on word

wlist = line1.lower().split()
#print(wlist)
punct = '"",.'
wtot = [word.strip(punct) for word in wlist]
#print(wtot)

wsetsorted = sorted(set(wtot))



worddict = { word:i for (i, word) in enumerate(wsetsorted) }
#print(wtot)
print(worddict)

wordvec = torch.zeros(len(wtot), len(worddict))

for i, w in enumerate(wtot):
    idx = worddict[w]
    wordvec[i][idx] = 1

#print(wlist)
print(wordvec)

#print(len(worddict), worddict['impossible'])

# broadcasting in pytorch

x = torch.ones(())
y = torch.ones(3,1)
z = torch.ones(1,3)
a = torch.ones(2, 1, 1)

print(f" z: {z.shape}, a: {a.shape}")
print("x * y:", x, y, x*y, (x * y).shape)
print("=========================")
print("y * z:", y*z, (y * z).shape)
print("=========================")
print("y * z * a:", y*z*a, (y * z * a).shape)

a1 = torch.tensor([[ 1], [ 2], [3]])
a2 = torch.ones(1,3)
print(a1.shape)
print(a1+a2)

a1 = torch.tensor([[ 1,4], [ 2,5], [3,6]])
a2 = torch.ones(3,2)
print(a1.shape)
print(a1+a2)