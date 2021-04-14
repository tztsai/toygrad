from init import *

a = Param(size=[5, 5])
b = np.sign(a)
c = np.ones([3, 4]).view(Param)
model = Model(Affine(4))
model(Param(size=[10, 3]))
pickle.dump(a, open('a', 'wb'))
model.save('tmp')
a_ = pickle.load(open('a', 'rb'))
model_ = Model.load('tmp')
for f in ['a', 'tmp']:
    os.remove(f)