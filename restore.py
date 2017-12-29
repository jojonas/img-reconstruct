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
    parser.add_argument("--low-quantile", help="Lower quantile to adjust curves to (0-1).", type=float, default=0.05)
    parser.add_argument("--high-quantile", help="Upper quantile to adjust curves to (0-1).", type=float, default=0.95)
    parser.add_argument("--low-target", help="Target value for pixels at the low quantile (0-1).", type=float, default=0.10)
    parser.add_argument("--high-target", help="Target value for pixels at the high quantile (0-1).", type=float, default=0.90)
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
    low_quantile = np.percentile(data, args.low_quantile*100)
    high_quantile = np.percentile(data, args.high_quantile*100)

    sections_x = [0, low_quantile, high_quantile, 1]
    sections_y = [0, args.low_target, args.high_target, 1]

    out = apply_section(data, sections_x, sections_y)
    return np.clip(out, 0, 1)

def apply_section(data, x_points, y_points):
    assert len(x_points) == len(y_points)

    out = np.zeros_like(data)

    for i in range(len(x_points) - 1):
        x_start = x_points[i]
        x_end = x_points[i+1]
        y_start = y_points[i]
        y_end = y_points[i+1]

        mask = np.logical_and(data > x_start, data <= x_end)

        slope = (y_end - y_start) / (x_end - x_start)
        out[mask] = (data[mask] - x_start) * slope + y_start

    return out


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
