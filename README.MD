Image-Restoration
=================

This script removes color tints from scans of aged negatives or diapositives.

Usage:

    usage: restore.py [-h] [--low LOW] [--high HIGH]
                                  [--pad-low PAD_LOW] [--pad-high PAD_HIGH]
                                  [--invert] [--multiprocessing] [--out OUT]
                                  filename [filename ...]

    Automagically reconstruct scans of old photographs.

    positional arguments:
      filename              File(s) to reconstruct.

    optional arguments:
      -h, --help            show this help message and exit
      --low LOW             Lower percentile to adjust curves to (0-1). (default:
                            0.1)
      --high HIGH           Upper percentile to adjust curves to (0-1). (default:
                            0.9)
      --pad-low PAD_LOW     Additional padding below the histogram (0-255).
                            (default: 20)
      --pad-high PAD_HIGH   Additional padding above the histogram (0-255).
                            (default: 30)
      --invert, -i          Invert colors (for negatives). (default: False)
      --multiprocessing, -m
                            Use multiple processors. (default: False)
      --out OUT, -o OUT     Directory to place reconstructed images into.
                            (default: None)

Dependencies:
* [Python 3](https://www.python.org/)
* [Python Pillow Module](https://python-pillow.org/)
* [NumPy](http://www.numpy.org/)
* optional: [piexif](https://pypi.python.org/pypi/piexif)

Examples
========

Here are some before / after comparisons of scanned diapositives.
They were created with the following command: `python restore.py examples\*.jpg`

![before](https://raw.githubusercontent.com/jojonas/img-reconstruct/master/examples/DSC_0840.JPG)
![after](https://raw.githubusercontent.com/jojonas/img-reconstruct/master/examples/DSC_0840_restored.JPG)

![before](https://raw.githubusercontent.com/jojonas/img-reconstruct/master/examples/DSC_0921.JPG)
![after](https://raw.githubusercontent.com/jojonas/img-reconstruct/master/examples/DSC_0921_restored.JPG)

![before](https://raw.githubusercontent.com/jojonas/img-reconstruct/master/examples/DSC_0972.JPG)
![after](https://raw.githubusercontent.com/jojonas/img-reconstruct/master/examples/DSC_0972_restored.JPG)

