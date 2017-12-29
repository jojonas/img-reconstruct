Image-Restoration
=================

This script removes color tints from scans of aged negatives or diapositives.

Usage:

    usage: restore.py [-h] [--low-quantile LOW_QUANTILE]
                      [--high-quantile HIGH_QUANTILE] [--low-target LOW_TARGET]
                      [--high-target HIGH_TARGET] [--invert] [--multiprocessing]
                      [--quality QUALITY] [--out OUT]
                      filename [filename ...]

    Automagically reconstruct scans of old photographs.

    positional arguments:
      filename              File(s) to reconstruct.

    optional arguments:
      -h, --help            show this help message and exit
      --low-quantile LOW_QUANTILE
                            Lower quantile to adjust curves to (0-1). (default:
                            0.05)
      --high-quantile HIGH_QUANTILE
                            Upper quantile to adjust curves to (0-1). (default:
                            0.95)
      --low-target LOW_TARGET
                            Target value for pixels at the low quantile (0-1).
                            (default: 0.1)
      --high-target HIGH_TARGET
                            Target value for pixels at the high quantile (0-1).
                            (default: 0.9)
      --invert, -i          Invert colors (for negatives). (default: False)
      --multiprocessing, -m
                            Use multiple processors. (default: False)
      --quality QUALITY     Set JPEG quality (0-100). (default: 95)
      --out OUT, -o OUT     Directory to place reconstructed images into.
                            (default: None)


Dependencies:
* [Python 3](https://www.python.org/)
* [Python Pillow Module](https://python-pillow.org/)
* [NumPy](http://www.numpy.org/)
* optional: [piexif](https://pypi.python.org/pypi/piexif)
* optional: [rawpy](https://pypi.python.org/pypi/rawpy)

Examples
========

Here are some before / after comparisons of scanned diapositives.
They were created with the following command: `python restore.py examples\*.jpg`

![before](https://raw.githubusercontent.com/jojonas/img-reconstruct/master/examples/DSC_0840.JPG)
![after](https://raw.githubusercontent.com/jojonas/img-reconstruct/master/examples/DSC_0840_restored.JPG)

![before](https://raw.githubusercontent.com/jojonas/img-reconstruct/master/examples/DSC_0921.JPG)
![after](https://raw.githubusercontent.com/jojonas/img-reconstruct/master/examples/DSC_0921_restored.JPG)

![before](https://raw.githubusercontent.com/jojonas/img-reconstruct/master/examples/DSC_0972.JPG)
![after](https://raw.githubusercontent.com/jojonas/img-reconstruct/master/examples/DSC_0972_restored.JPG)

