from matplotlib.image import imread
from my_utils.utils import interact
import sys

sys.path.append('..')

from init import *

img = imread('data/mario.jpg')  # RGB
img = img.mean(axis=-1)    # gray

conv = node.Conv2D(4, fan_in=img.shape)
conv.setup()

blur = np.array([[.1, .1, .1],
                 [.1, .2, .1],
                 [.1, .1, .1]])
blurry_img = conv.convolve(img, blur)

sharp = np.array([[-.1, -.1, -.1],
                  [-.1, 1.8, -.1],
                  [-.1, -.1, -.1]])
sharp_img = conv.convolve(img, sharp)

edgy = np.array([[.1, 0, -.1],
                 [.2, 0, -.2],
                 [.1, 0, -.1]])
edgy_img1 = conv.convolve(img, edgy)
edgy_img2 = conv.convolve(img, edgy.T)

conv.filters = node.Parameter([blur, sharp, edgy, edgy.T])

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.subplot(1, 3, 2)
plt.imshow(blurry_img, cmap='gray')
# plt.imshow(edgy_img1, cmap='gray')
plt.subplot(1, 3, 3)
plt.imshow(sharp_img, cmap='gray')
# plt.imshow(edgy_img2, cmap='gray')
plt.show()

blurry_img, sharp_img, edgy_img1, edgy_img2 = conv(img, batch=False)

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.subplot(1, 3, 2)
plt.imshow(blurry_img, cmap='gray')
# plt.imshow(edgy_img1, cmap='gray')
plt.subplot(1, 3, 3)
plt.imshow(sharp_img, cmap='gray')
# plt.imshow(edgy_img2, cmap='gray')
plt.show()

# interact()
