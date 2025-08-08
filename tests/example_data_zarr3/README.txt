This OME-Zarr file was downloaded from https://idr.github.io/ome-ngff-samples/ on July 31, 2025.
It is distributed under the CC-BY 4.0 license.

To download the file:
aws s3 sync --endpoint-url https://uk1s3.embassy.ebi.ac.uk --no-sign-request s3://idr/zarr/v0.5/idr0062A/6001240_labels.zarr 6001240_labels_v0.5.zarr

We only retain the lowest resolution (pyramid level 2), by removing the folders corresponding to pyramid levels 0 and 1, and deleting the corresponding coordinateTransformations from the zarr.json file. 