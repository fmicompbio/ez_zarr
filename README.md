# ez_zarr

## Goals
The aim of `ez_zarr` is to provide easy, high-level access
to OME-Zarr filesets (high content screening microscopy data, stored
according to the [NGFF](https://ngff.openmicroscopy.org/latest/)
specifications in OME-Zarr with additional metadata fields, for
example the ones generated by the [Fractal](https://fractal-analytics-platform.github.io/fractal-tasks-core/) platform).

The goal is that users can write simple scripts working with plates,
wells and fields of view, without having to understand how these
are represented within an OME-Zarr fileset.

## Example
```
# import module
from ez_zarr import zarr_wraps

# create `plate_3d` object representing an OME-Zarr fileset
plate_3d = zarr_wraps.FmiZarr('tests/example_data/plate_ones.zarr')

# print fileset summary
plate_3d
# FmiZarr plate_ones.zarr with 1 well and 2 channels
#   path: tests/example_data/plate_ones.zarr
#   n_wells: 1
#   n_channels: 2 (some-label-1, some-label-2)
#   n_pyramid_levels: 3
#   pyramid_zyx_scalefactor: [1. 2. 2.]
#   full_resolution_zyx_spacing: [1.0, 0.1625, 0.1625]
#   segmentations: 
#   tables (measurements): FOV_ROI_table
```

## Software status
[![unit-tests](https://github.com/fmi-basel/gbioinfo-ez_zarr/actions/workflows/test_and_deploy.yaml/badge.svg)](https://github.com/fmi-basel/gbioinfo-ez_zarr/actions/workflows/test_and_deploy.yaml)
[![codecov](https://codecov.io/gh/fmi-basel/gbioinfo-ez_zarr/graph/badge.svg?token=GEBLX8ENJ1)](https://codecov.io/gh/fmi-basel/gbioinfo-ez_zarr)

## Contributors and License
`ez_zarr` is released under the MIT License, and the copyright
is with the Friedrich Miescher Insitute for Biomedical Research
(see [LICENSE](https://github.com/fmi-basel/gbioinfo-ez_zarr/blob/main/LICENSE)).

`ez_zarr` is being developed at the Friedrich Miescher Institute for
Biomedical Research by [@silvbarb](https://github.com/silvbarb) and [@mbstadler](https://github.com/mbstadler).