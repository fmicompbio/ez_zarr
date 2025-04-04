# __main__.py

import sys
import os

import zarr
from ez_zarr import ome_zarr

def usage():
    print('ez_zarr version ' + ome_zarr.__version__ + '\n' +
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

            try:
                zarr_group = zarr.open_group(store=path, mode='r')
            except Exception as e:
                print(f"failed to find a zarr group in {path}")
                sys.exit(1)

            if len([x for x in zarr_group.group_keys() if x not in ['labels', 'tables']]) > 0:
                try:
                    obj = ome_zarr.import_plate(path)
                except Exception as e:
                    print(f"failed to import plate from {path} - are you pointing at the wrong folder?")
                    sys.exit(1)

            else:
                obj = ome_zarr.Image(path)

            print(obj)
        else:
            usage()

    # if no path is given, show usage
    else:
        usage()

if __name__ == "__main__":
    main()
