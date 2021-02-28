import torch


t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]

celcius = torch.tensor(t_c)
unknown = torch.tensor(t_u)
unknown = 0.1 * unknown   # approx normalization between -1 and 1 

y = celcius

# w - weights, b- bias
# unknown scale is the observation

def model(unk, w, b):
    return w * unk + b

def lossfn(pred, y):
    sqdiff = (pred - y)**2
    return sqdiff.mean()

# initialize weight and bias

w = torch.ones(())
b = torch.zeros(())


pred = model(unknown, w, b)
loss = lossfn(pred, y)
print(loss)

def dloss(pred, y):
    diff = 2 * (pred - y) / pred.size(0)
    return diff

# return gradients
def gradfn(unk, y, pred, w, b):
    diff = dloss(pred,y)
    dw = diff * unk
    db = diff * 1.0
    return torch.stack([dw.sum(), db.sum()])

def training_loop(epochs, lr, params, unk, y):
    for epoch in range(1, epochs + 1):
        w, b = params
        pred = model(unk, w, b)
        loss = lossfn(pred, y)
        grad = gradfn(unk, y, pred, w, b)
        params = params - lr * grad
        print('Epoch %d, Loss %f' % (epoch, float(loss)))
        print('params:', params)
        print('grads:', grad)
    return params

#training_loop(epochs=100, lr=.01, params=torch.tensor([1.0, 0.0]), unk=unknown, y =y)

params = torch.tensor([1.0, 0.0], requires_grad=True)

def training_loop_grad(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        if params.grad is not None:
            params.grad.zero_()

        t_p = model(t_u, params[0], params[1])
        loss = lossfn(t_p, t_c)
        loss.backward()

        with torch.no_grad():
            params -= learning_rate * params.grad
        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
    return params

outparms = training_loop_grad(n_epochs=5000, learning_rate=.01, params=params, t_u=unknown, t_c =y)

print(outparms)