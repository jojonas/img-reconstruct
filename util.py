import os, os.path
import warnings

import numpy as np
from PIL import Image

# from https://en.wikipedia.org/wiki/Raw_image_format
RAW_EXTENSIONS = [
    ".3fr",
    ".ari", ".arw",
    ".bay",
    ".crw", ".cr2",
    ".cap",
    ".data", ".dcs", ".dcr", ".dng",
    ".drf",
    ".eip", ".erf",
    ".fff",
    ".gpr",
    ".iiq",
    ".k25", ".kdc",
    ".mdc", ".mef", ".mos", ".mrw",
    ".nef", ".nrw",
    ".obm", ".orf",
    ".pef", ".ptx", ".pxn",
    ".r3d", ".raf", ".raw", ".rwl", ".rw2", ".rwz",
    ".sr2", ".srf", ".srw",
    ".tif",
    ".x3f"
]

def load_image(filename):
    ext = os.path.splitext(filename)[1].lower()

    if ext in RAW_EXTENSIONS:
        try:
            import rawpy
        except ImportError:
            warnings.warn("Python module 'rawpy' is required in order to process RAW images.")
        else:
            raw = rawpy.imread(filename)
            params = rawpy.Params(output_bps=16)
            rgb = raw.postprocess(params)

            array = rgb.astype(float)/ 2**16
            return array, None

    image = Image.open(filename)
    array = np.array(image, dtype=np.uint8).astype(float) / 255
    return array, image.info

def save_image(filename, array, **kwargs):
    assert filename.lower().endswith('.jpg')

    rgb = np.clip(np.floor(array*255), 0, 255).astype(np.uint8)
    image = Image.fromarray(rgb)
    image.save(filename, format='JPEG', subsampling=0, **kwargs)
