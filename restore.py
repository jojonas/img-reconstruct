import argparse
import glob
import multiprocessing
import os, os.path
import warnings

import numpy as np
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(
        description="Automagically reconstruct scans of old photographs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("filename", help="File(s) to reconstruct.", nargs="+")
    parser.add_argument("--low", help="Lower percentile to adjust curves to (0-1).", type=float, default=0.10)
    parser.add_argument("--high", help="Upper percentile to adjust curves to (0-1).", type=float, default=0.90)
    parser.add_argument("--pad-low", help="Additional padding below the histogram (0-255).", type=float, default=20)
    parser.add_argument("--pad-high", help="Additional padding above the histogram (0-255).", type=float, default=30)
    parser.add_argument("--invert", "-i", help="Invert colors (for negatives).", action="store_true")
    parser.add_argument("--multiprocessing", "-m", help="Use multiple processors.", action="store_true")
    parser.add_argument("--out", "-o", help="Directory to place reconstructed images into.", default=None)

    return parser.parse_args()

def main():
    args = parse_args()

    mp_args = [(filename, args) for path in args.filename for filename in glob.glob(path)]

    if args.multiprocessing:
        pool = multiprocessing.Pool()
        pool.map(process, mp_args)
    else:
        for tuple in mp_args:
            process(tuple)

def process(params):
    filename, args = params

    if args.out:
        outname = join_out_filename(args.out, os.path.basename(filename))
    else:
        outname = compute_out_filename(filename)

    print("Processing", filename, "=>", outname)

    image = load_image(filename)

    restored_image = restore(image, args)

    # transfer exif info
    try:
        import piexif
    except ImportError:
        warnings.warn("Python module 'piexif' is required in order to preserve EXIF information.")
        exif_bytes = b''
    else:
        exif_dict = piexif.load(image.info['exif'])
        del exif_dict['thumbnail']
        exif_bytes = piexif.dump(exif_dict)

    save_image(outname, restored_image, quality=95, exif=exif_bytes)

    image.close()

def plot_histogram(image):
    import matplotlib.pyplot as plt
    channels = image.split()
    for channel, color in zip(channels, ('red', 'green', 'blue')):
        data = np.asarray(channel, dtype=np.uint8)
        plt.hist(data.flatten(), range=(0, 255), histtype='step', bins=50, normed=True, color=color)
    plt.show()

def load_image(filename):
    return Image.open(filename)

def restore(image, args):
    array = np.array(image, dtype=np.uint8)
    if args.invert:
        array = 255 - array

    for channel in range(array.shape[-1]):
        data = array[:,:,channel]
        data = np.floor(255 * restore_channel(data, args))
        array[:,:,channel] = data

    return Image.fromarray(array)

def restore_channel(data, args):
    low_value = np.percentile(data, args.low*100) - args.pad_low
    high_value = np.percentile(data, args.high*100) + args.pad_high
    range = high_value - low_value
    return np.clip((data-low_value)/range, 0, 1)

def compute_out_filename(filename):
    name, ext = os.path.splitext(filename)
    name += "_restored"
    return name + ext

def join_out_filename(out, filename):
    if not os.path.exists(out):
        os.makedirs(out)
    return os.path.join(out, os.path.basename(filename))

def save_image(filename, image, **kwargs):
    assert filename.lower().endswith('.jpg')
    image.save(filename, format='JPEG', subsampling=0, **kwargs)

if __name__=="__main__":
    main()