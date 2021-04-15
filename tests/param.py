from init import *

a = Param(size=[5, 5])

print(a.training)
with Param.not_training():
    print(a.training)
print(a.training)

b = np.sign(a)
c = np.ones([3, 4]).view(Param)

model = Model(Affine(4))
model(Param(size=[10, 3]))
model.save('tmp')
model_ = Model.load('tmp')

for f in ['tmp']: os.remove(f)