# __main__.py

import sys
import os

from ez_zarr import hcs_wrappers

def usage():
    print('ez_zarr version ' + hcs_wrappers.__version__ + '\n' +
          'usage:\n  python -m ez_zarr path\n' +
          'where `path` is a path to an OME-Zarr output from Fractal\n' +
          'or a folder containing such outputs.\n')

def main():
    """Show info on Fractal OME-Zarr fileset"""

    # if a path is given, show information on the fileset
    if len(sys.argv) > 1:
        path = sys.argv[1]

        if os.path.isdir(path):
            path = os.path.normpath(path)
            if path[-5:] == '.zarr':
                obj = hcs_wrappers.FractalZarr(path)
            else:
                obj = hcs_wrappers.FractalZarrSet(path)
            print(obj)
        else:
            usage()

    # if no path is given, show usage
    else:
        usage()

if __name__ == "__main__":
    main()
