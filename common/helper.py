import datetime

def training_loop(epochs, optimizer, model, loss_fn, trainX, trainY, valX, valY):
    for epoch in range(1, epochs + 1):
        pred = model(trainX)
        train_loss = loss_fn(pred, trainY)

        pval = model(valX)
        val_loss = loss_fn(pval, valY)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 1000 == 0:
            #print(trainX, pred)
            print(f"Epoch {epoch}, Training loss {train_loss.item():.4f}," f" Validation loss {val_loss.item():.4f}")


def softmax(x):
    return torch.exp(x) / torch.exp(x).sum()


def trainloop(epochs, optimizer, model, loss_fn, train_loader):

    for epoch in range(1, epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:

            output = model(imgs)
            loss = loss_fn(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_train / len(train_loader)))

