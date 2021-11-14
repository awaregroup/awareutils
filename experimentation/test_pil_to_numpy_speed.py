import timeit

import numpy as np
from PIL import Image

img = Image.new("RGB", (2000, 2000), (0, 0, 0))


def to_numpy(im):
    im.load()
    # unpack data
    e = Image._getencoder(im.mode, "raw", im.mode)
    e.setimage(im.im)

    # NumPy buffer for the result
    shape, typestr = Image._conv_type_shape(im)
    data = np.empty(shape, dtype=np.dtype(typestr))
    mem = data.data.cast("B", (data.data.nbytes,))

    bufsize, s, offset = 65536, 0, 0
    while not s:
        l, s, d = e.encode(bufsize)
        mem[offset : offset + len(d)] = d
        offset += len(d)
    if s < 0:
        raise RuntimeError("encoder error %d in tobytes" % s)
    return data


print("to_numpy")
print(timeit.timeit("to_numpy(img)", number=10, globals=globals()))
print("np.array")
print(timeit.timeit("np.array(img)", number=10, globals=globals()))
print("np.asarray")
print(timeit.timeit("np.asarray(img)", number=10, globals=globals()))
