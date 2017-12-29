import argparse
import glob
import multiprocessing
import os, os.path
import warnings

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from util import load_image

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
    array = load_image(filename)

    channels = array.shape[-1]
    for channel, color in zip(range(channels), ('red', 'green', 'blue')):
        data = array[:,:,channel]
        plt.hist(data.flatten(), range=(0, 1), histtype='step', bins=255, normed=True, color=color)

    plt.xlim(0, 1)
    plt.xlabel("Value")
    plt.show()

if __name__=="__main__":
    main()
