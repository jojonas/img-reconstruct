from __future__ import print_function

import argparse
import glob
import multiprocessing
import os, os.path
import warnings

import numpy as np
from PIL import Image

from util import load_image, save_image

def parse_args():
    parser = argparse.ArgumentParser(
        description="Automagically reconstruct scans of old photographs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("filename", help="File(s) to reconstruct.", nargs="+")
    parser.add_argument("--low", help="Lower percentile to adjust curves to (0-1).", type=float, default=0.10)
    parser.add_argument("--high", help="Upper percentile to adjust curves to (0-1).", type=float, default=0.90)
    parser.add_argument("--pad-low", help="Additional padding below the histogram (0-1).", type=float, default=0.05)
    parser.add_argument("--pad-high", help="Additional padding above the histogram (0-1).", type=float, default=0.05)
    parser.add_argument("--invert", "-i", help="Invert colors (for negatives).", action="store_true")
    parser.add_argument("--multiprocessing", "-m", help="Use multiple processors.", action="store_true")
    parser.add_argument("--quality", help="Set JPEG quality (0-100).", type=int, default=95)
    parser.add_argument("--out", "-o", help="Directory to place reconstructed images into.", default=None)

    return parser.parse_args()

def main():
    args = parse_args()

    mp_args = [(filename, args) for path in args.filename for filename in glob.glob(path)]

    if args.multiprocessing:
        pool = multiprocessing.Pool()
        pool.map(process, mp_args)
        pool.close()
    else:
        for tuple in mp_args:
            process(tuple)

def process(params):
    # unpack arguments (needed for multiprocessing)
    filename, args = params

    if args.out:
        outname = join_out_filename(args.out, os.path.basename(filename))
    else:
        outname = compute_out_filename(filename)

    print("Processing", filename, "=>", outname)

    array, info = load_image(filename)

    restored_image = restore(array, args)

    if info:
        # transfer exif info
        try:
            import piexif
        except ImportError:
            warnings.warn("Python module 'piexif' is required in order to preserve EXIF information.")
            exif_bytes = b''
        else:
            exif_dict = piexif.load(info['exif'])
            # remove thumbnail from exif info (new appearance)
            del exif_dict['thumbnail']
            exif_bytes = piexif.dump(exif_dict)
    else:
        exif_bytes = b''

    save_image(outname, restored_image, quality=args.quality, exif=exif_bytes)

def restore(array, args):
    if args.invert:
        array = 1 - array

    for channel in range(array.shape[-1]):
        data = array[:,:,channel]
        data = restore_channel(data, args)
        array[:,:,channel] = data

    return array

def restore_channel(data, args):
    # percentile expects quantile value between 0 and 100
    low_value = np.percentile(data, args.low*100) - args.pad_low
    high_value = np.percentile(data, args.high*100) + args.pad_high

    range = high_value - low_value
    data = (data-low_value)/range

    # restrain output data to [0, 1] (can become negative otherwise)
    return np.clip(data, 0, 1)

def compute_out_filename(filename):
    name, ext = os.path.splitext(filename)
    name += "_restored"
    return name + ".jpg"

def join_out_filename(out, filename):
    if not os.path.exists(out):
        os.makedirs(out)
    return os.path.join(out, os.path.basename(filename))

if __name__=="__main__":
    main()
