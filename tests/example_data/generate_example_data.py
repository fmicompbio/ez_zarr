# this file is based on https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/main/tests/data/generate_zarr_ones.py
# to generate test data, run the following from the example_data directory
#     python generate_example_data.py

import json
import os
import shutil

import dask.array as da
import numpy as np
import pandas as pd
import zarr
from anndata._io.specs import write_elem
import itertools

from fractal_tasks_core.roi import prepare_FOV_ROI_table

# gloabl parameters
axes = [
    {"name": "c", "type": "channel"},
    {"name": "z", "type": "space", "unit": "micrometer"},
    {"name": "y", "type": "space", "unit": "micrometer"},
    {"name": "x", "type": "space", "unit": "micrometer"},
]

component = "B/03/0/" # well-row/well-column/0
plate_zattrs = {
    'plate': {
        'acquisitions': [{'id': 0, 'name': 'my-experiment'}],
        'columns': [{'name': '03'}],
        'rows': [{'name': 'B'}],
        'wells': [{'columnIndex': 0, 'path': 'B/03', 'rowIndex': 0}]
    }
}
table_zattrs = {
    'tables': ['FOV_ROI_table']
}
label_zattrs = {'labels': ['organoids']}

pxl_z = 1.0      # voxel size in z
pxl_y = 0.1625   # voxel size in y
pxl_x = 0.1625   # voxel size in x

size_x = 640     # field of view size in x
size_y = 540     # field of view size in y
size_z = 3       # z pixel depth

cxy = 2          # scaling factor between sequential pyramid levels
num_levels = 3   # number of pyramid levels

# fileset specific parameters (lists)
num_C = [2, 2]        # number of channels
num_Z = [3, 1]        # number planes in Z
num_Y = [2, 2]        # number of fields of view in Y
num_X = [2, 2]        # number of fields of view in X

zarrurl = ["plate_ones.zarr/",
           "plate_ones_mip.zarr/"]

create_labels = [False, True] # should 'labels' be generated

# generate filesets
for i in range(len(zarrurl)):
    x = da.ones(
        (num_C[i], num_Z[i], num_Y[i] * size_y, num_X[i] * size_x), chunks=(1, 1, size_y, size_x)
    ).astype(np.uint16)

    if os.path.isdir(zarrurl[i]):
        shutil.rmtree(zarrurl[i])

    for ind_level in range(num_levels):
        scale = 2**ind_level
        y = da.coarsen(np.min, x, {2: scale, 3: scale}).rechunk(
            (1, 1, size_y, size_x), balance=True
        )
        y.to_zarr(
            zarrurl[i], component=f"{component}{ind_level}", dimension_separator="/"
        )

    with open(f"{zarrurl[i]}.zattrs", "w") as jsonfile:
        json.dump(plate_zattrs, jsonfile, indent=4)

    cT = "coordinateTransformations"
    zattrs = {
        "multiscales": [
            {
                "axes": axes,
                "datasets": [
                    {
                        "path": level,
                        cT: [
                            {
                                "type": "scale",
                                "scale": [
                                    1.0,
                                    pxl_z,
                                    pxl_y * cxy**level,
                                    pxl_x * cxy**level,
                                ],
                            }
                        ],
                    }
                    for level in range(num_levels)
                ],
                "version": "0.4",
            }
        ],
        "omero": {
            "channels": [
                {
                    "wavelength_id": "A01_C01",
                    "label": "some-label-1",
                    "window": {"min": "0", "max": "10", "start": "0", "end": "10"},
                    "color": "00FFFF",
                },
                {
                    "wavelength_id": "A01_C02",
                    "label": "some-label-2",
                    "window": {"min": "0", "max": "10", "start": "0", "end": "10"},
                    "color": "00FFFF",
                },
            ]
        },
    }
    with open(f"{zarrurl[i]}{component}.zattrs", "w") as jsonfile:
        json.dump(zattrs, jsonfile, indent=4)

    df_nrow = num_X[i] * num_Y[i] * num_Z[i]
    df = pd.DataFrame(np.zeros((df_nrow, 8)), dtype=int)
    df.index = [str(j) for j in range(df_nrow)]
    df.columns = [
        "x_micrometer",
        "y_micrometer",
        "z_micrometer",
        "len_x_micrometer",
        "len_y_micrometer",
        "len_z_micrometer",
        "x_micrometer_original",
        "y_micrometer_original",
    ]
    x_mu = []
    y_mu = []
    z_mu = []
    for zyx in itertools.product([j * size_z for j in range(num_Z[i])],
                                 [j * size_y for j in range(num_Y[i])],
                                 [j * size_x for j in range(num_X[i])]):
        z_mu.append(zyx[0])
        y_mu.append(zyx[1])
        x_mu.append(zyx[2])
    df["x_micrometer"] = [el * pxl_x for el in x_mu]
    df["y_micrometer"] = [el * pxl_y for el in y_mu]
    df["z_micrometer"] = [el * pxl_z for el in z_mu]
    df["x_pixel"] = [size_x] * df_nrow
    df["y_pixel"] = [size_y] * df_nrow
    df["z_pixel"] = [size_z] * df_nrow
    df["pixel_size_x"] = [pxl_x] * df_nrow
    df["pixel_size_y"] = [pxl_y] * df_nrow
    df["pixel_size_z"] = [pxl_z] * df_nrow
    df["bit_depth"] = [16.0] * df_nrow


    FOV_ROI_table = prepare_FOV_ROI_table(df)
    print(FOV_ROI_table.to_df())

    group_tables = zarr.group(f"{zarrurl[i]}{component}/tables")
    write_elem(group_tables, "FOV_ROI_table", FOV_ROI_table)
    with open(f"{zarrurl[i]}{component}tables/.zattrs", "w") as jsonfile:
        json.dump(table_zattrs, jsonfile, indent=4)
    
    if create_labels[i]:
        group_labels = zarr.group(f"{zarrurl[i]}{component}/labels")
        for ind_level in range(num_levels):
            scale = 2**ind_level
            y = da.coarsen(np.min, x, {2: scale, 3: scale}).rechunk(
                (1, 1, size_y, size_x), balance=True
            )
            y.to_zarr(
                zarrurl[i], component=f"{component}/labels/organoids/{ind_level}", dimension_separator="/"
            )
        with open(f"{zarrurl[i]}{component}labels/.zattrs", "w") as jsonfile:
            json.dump(label_zattrs, jsonfile, indent=4)

