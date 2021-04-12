from init import *
from data.mnist import *


def classifier(classes):
    return Compose(
        Affine(32), dropout(0.2), ReLU,
        # Affine(16), dropout(0.2), ReLU,
        Affine(classes)
    )
    
def classifier2(classes):
    return Compose(
        Conv2D(32, 5, stride=2), ReLU,
        Conv2D(64, 5, stride=2), ReLU,
        Conv2D(128, 3, stride=2), ReLU,
        flatten, Affine(classes)
    )

clf = classifier(10)

# x_train, x_test = x_train.reshape(-1,1,*im_size), x_test.reshape(-1,1,*im_size)
y_train, y_test = [onehot(y, 10) for y in (y_train, y_test)]
(x_train, y_train), (x_val, y_val) = train_val_split(x_train, y_train)


def accuracy(out, labels):
    return (np.argmax(out, axis=1) == np.argmax(labels, axis=1)).astype(float).mean()

history = clf.fit(x_train, y_train, epochs=10, loss='smce',
                  val_data=(x_val,y_val), metrics={'val_acc': accuracy})


# smartie = toybrain(input_dim, 10)
# batches = BatchLoader(x_train, y_train)

# for epoch in range(30):
#     print(f'{epoch=}')
#     losses = []
#     for x, y in batches:
#         mistake = smartie(x, y)
#         pars = mistake.backward()
#         losses.append(mistake)
#         for p in pars:
#             p -= 5e-3 * p.grad
#             p.zero_grad()
#     print('loss=%.2f' % np.mean(losses))
#     with Param.not_training:
#         print('acc=%.1f\%' % accuracy(smartie))
