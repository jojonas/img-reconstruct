import os, os.path
import warnings

import numpy as np
from PIL import Image

def load_image(filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext in (".nef", ):
        try:
            import rawpy
        except ImportError:
            warnings.warn("Python module 'rawpy' is required in order to process RAW images.")
        raw = rawpy.imread(filename)
        params = rawpy.Params(output_bps=16)
        rgb = raw.postprocess(params)

        array = rgb.astype(float)/ 2**16
        info = None
    else:
        image = Image.open(filename)

        array = np.array(image, dtype=np.uint8).astype(float) / 255

        info = image.info

    return array, info

def save_image(filename, array, **kwargs):
    assert filename.lower().endswith('.jpg')

    rgb = np.clip(np.floor(array*255), 0, 255).astype(np.uint8)
    image = Image.fromarray(rgb)
    image.save(filename, format='JPEG', subsampling=0, **kwargs)
