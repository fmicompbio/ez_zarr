# Load libraries
from ez_zarr import ome_zarr
from ngio import create_ome_zarr_from_array, open_ome_zarr_container
import tifffile as tf
import numpy as np
from skimage import feature, segmentation, morphology

# Download example data from https://docs.openmicroscopy.org/ome-model/5.6.3/ome-tiff/data.html
# curl -o tubhiswt-4D.zip https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/tubhiswt-4D.zip
# unzip tubhiswt-4D.zip

# Open tiff file as numpy array
x = tf.imread("tubhiswt-4D/tubhiswt_C0_TP7.ome.tif")

# Subset to tczyx shape (2, 1, 2, 512, 512)
xsub = x[0:1, slice(10, 13), 0:2, 0:512, 0:512] # ctzyx
xsub = np.transpose(xsub, (1, 0, 2, 3, 4)) # reorder to tczyx

# Create OME-Zarr
new_ome_zarr_image = create_ome_zarr_from_array(
    store="tubhiswt4D_sub.zarr", 
    array=xsub, 
    xy_pixelsize=1.0, 
    z_spacing=1.0,
    time_spacing=1.0,
    levels = 3,
    xy_scaling_factor = 2.0,
    z_scaling_factor = 1.0,
    space_unit = "micrometer",
    time_unit = "seconds", # "second" ?
    axes_names = ["t", "c", "z", "y", "x"],
    channel_labels = None,
    channel_wavelengths = None,
    percentiles = (0.1, 99.9),
    channel_colors = None,
    channel_active = None,
    name = "Tubhiswt-4D",
    chunks = None,
    overwrite = True
)

# Create labels manually with napari (out-of-the-box automatic segmentation via 
# thresholding or edge detection didn't work well in this case)
# drag-and-drop the .zarr into napari
# create a 'New labels layer'
# activate the layer, click on the paintbrush in the toolbar and draw the outline
# use the fill tool to fill the outline
# either redo this for each T/Z, or use the console in napari to copy the data from
# one layer to the others
# access the labels in the console via viewer.layers[1].data
# File -> Save Selected Layers -> save as labels_from_napari.tif

# Read labels generated with napari, and add to image
label = tf.imread("labels_from_napari.tif")
new_label = new_ome_zarr_image.derive_label("embryo", overwrite=True)
new_label.set_array(label)

# consolidate across all pyramid levels
new_label.consolidate()

# done - try reading and plotting with ez-zarr
img = ome_zarr.Image("tubhiswt4D_sub.zarr")
img.plot(time_index=0, label_name="embryo")

if __name__ == '__main__':
    pass

