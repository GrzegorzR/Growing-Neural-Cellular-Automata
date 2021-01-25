import numpy as np
from PIL import Image

def unpad(arr, dim_sizes):
    for dim, s in dim_sizes.items():
        arr = np.delete(arr, np.s_[-s:], axis=dim)
        arr = np.delete(arr, np.s_[0:s], axis=dim)
    return arr

def replicate_edges(arr, dim_sizes):
    arr = unpad(arr, dim_sizes)
    pads = []
    for i in range(arr.ndim):
        if i in dim_sizes:
            pads.append((dim_sizes[i], dim_sizes[i]))
        else:
            pads.append((0,0))
    arr = np.pad(arr, pads, mode='wrap')
    return arr


if __name__ == '__main__':
    img_path = 'data/pol.jpg'
    out_path = 'test_wrap.png'
    arr = np.array(Image.open(img_path))
    arr = replicate_edges(arr, {0:30,1:30})

    img = Image.fromarray(arr)
    img.save(out_path)
