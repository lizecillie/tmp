from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from _growcut import growcut
from skimage import io, img_as_float, transform, draw, filter, color
from skimage.filter import sobel
from skimage.io import ImageCollection, imsave
from skimage.transform import resize
from StringIO import StringIO


def orientation(A):
    A = color.rgb2gray(A)
    edge_A = filter.sobel(A)
    r = 
    c = 
    # Find principle axes of the coordinates
    C = np.array((r, c)).T
    u = C.mean(axis=0)
    C = C - u
    A = C.T.dot(C)

    w, v = np.linalg.eig(A)

    w = np.abs(w)
    ix = np.argsort(w)[::-1]

    w = w[ix]
    v = v[:, ix]

    theta = np.arctan2(v[0, 0], v[1, 0])
    if (0 < (theta % np.pi) < np.pi / 2):
        return 'Left'
    else:
        return 'Right'


ic = ImageCollection('*.jpg')
s = StringIO()

for n, image in enumerate(ic):
    image = resize(image,(256, 256))
    print "Image number %d is %s" % (n, image)
    
    state = np.zeros((image.shape[0], image.shape[1], 2))

    if orientation(image)==Left:
        foreground_pixels = np.array([(35, 90), (75, 90), (115, 90), (160, 90), (74, 140), (160, 140), (160, 210), (115, 170)])
        background_pixels = np.array([(20, 25), (65, 25), (110, 25), (170, 25), (20, 170), (20, 240), (80, 240), (60, 210)])
    else:
        foreground_pixels = np.array([(40, 155), (90, 155), (140, 155), (180, 155), (180, 100), (180, 45), (140, 70), (90, 100)])
        background_pixels = np.array([(25, 215), (80, 215), (140, 215), (200, 215), (25, 20), (25, 70), (100, 20), (55, 45)]) 

    for (r, c) in background_pixels:
        state[r, c] = (0, 1)

    for (r, c) in foreground_pixels:
        state[r, c] = (1, 1)

    out = growcut(image, state, window_size=5, max_iter=200)
    
    import matplotlib.pyplot as plt

    f, (ax0, ax1) = plt.subplots(1, 2, figsize=(7, 3))

    ax0.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
    ax0.plot(foreground_pixels[:, 1], foreground_pixels[:, 0],
             color='blue', marker='o', linestyle='none', label='Foreground')
    ax0.plot(background_pixels[:, 1], background_pixels[:, 0],
             color='red', marker='o', linestyle='none', label='Background')
    ax0.set_title('Input image')
    ax0.axis('image')

    ax1.imshow(out[..., None] * image, interpolation='nearest', cmap=plt.cm.gray, vmin=0, vmax=1)
    ax1.set_title('Foreground / background')
    ax0.axis('image')


    inumber = '%d' %(n)
    imsave(inumber, out)
    #plt.savefig('demo1.png')

plt.show()


