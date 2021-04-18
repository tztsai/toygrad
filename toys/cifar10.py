import pickle
from toych import *


(x_tr, y_tr), (x_ts, y_ts) = pickle.load(open('../data/cifar-10.pkl', 'rb'), encoding='bytes')
x_tr, x_ts = [x.reshape(-1, 3, 32, 32) for x in [x_tr, x_ts]]
(x_tr, y_tr), (x_va, y_va) = train_val_split(x_tr, y_tr)


def CNN1(classes):
    ker1 = Param(size=[32, 3, 5, 5])
    ker2 = Param(size=[64, 32, 3, 3])
    fc = Affine(classes)
    
    def forward(imgs):
        imgs = Param(imgs, kind='constant')
        h1 = imgs.conv2d(ker1, stride=2).relu()
        h2 = h1.conv2d(ker2, stride=2).relu()
        return fc(h2.flatten()).softmax()
        
    return Model(forward)

def CNN2(classes):
    return Compose(
        Conv2D(32, size=5, stride=2), ReLU,
        Conv2D(64, size=3, stride=2), ReLU,
        flatten, Affine(classes), softmax
    )

class CNN3(Model):
    def __init__(self, classes):
        self.conv1 = Conv2D(32, 5, stride=2)
        self.conv2 = Conv2D(64, 5, stride=2)
        self.fc = Affine(classes)
        
    def apply(self, imgs):
        h1 = self.conv1(imgs).relu()
        h2 = self.conv2(h1).relu()
        return self.fc(h2.flatten()).softmax()


clf1, clf2, clf3 = CNN1(10), CNN2(10), CNN3(10)

# for clf in [clf1, clf2, clf3]:
#     clf1.fit(x_tr, y_tr, epochs=2, loss='crossentropy',
#              val_data=(x_va, y_va), metrics={'val_acc': accuracy})

