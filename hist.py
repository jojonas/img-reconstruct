import argparse
import glob
import multiprocessing
import os, os.path
import warnings

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot histogram of color channels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("filename", help="File to use.")

    return parser.parse_args()

def main():
    args = parse_args()
    plot_histogram(args.filename)

def plot_histogram(filename):
    image = Image.open(filename)

    channels = image.split()
    for channel, color in zip(channels, ('red', 'green', 'blue')):
        data = np.asarray(channel, dtype=np.uint8)
        plt.hist(data.flatten(), range=(0, 255), histtype='step', bins=50, normed=True, color=color)

    plt.xlim(0, 255)
    plt.xlabel("Value")
    plt.show()

if __name__=="__main__":
    main()
