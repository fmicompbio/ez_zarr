---
title: 'ez-zarr: A Python package for easy access and visualisation of OME-Zarr filesets'
tags:
  - Python
  - OME-Zarr
  - image file formats
  - image visualisation
authors:
  - name: Silvia Barbiero
    orcid: 0000-0001-7177-6947
    equal-contrib: true
    affiliation: "1, 2"
  - name: Charlotte Soneson
    orcid: 0000-0003-3833-2169
    equal-contrib: true
    affiliation: "1, 3"
  - name: Prisca Liberali
    orcid: 0000-0003-0695-6081
    affiliation: "1, 2, 4"
  - name: Michael B. Stadler
    orcid: 0000-0002-2269-4934
    equal-contrib: true
    affiliation: "1, 2, 3"
    corresponding: true 
affiliations:
 - name: Friedrich Miescher Institute for Biomedical Research, Basel, Switzerland
   index: 1
   ror: 01bmjkv45
 - name: University of Basel, Basel, Switzerland
   index: 2
   ror: 02s6k3f65
 - name: SIB Swiss Institute of Bioinformatics, Basel, Switzerland
   index: 3
   ror: 002n09z45
 - name: ETH Zürich, Department for Biosystems Science and Engineering (D-BSSE), Basel, Switzerland
   index: 4
   ror: 05a28rw58
date: 06 February 2025
bibliography: paper.bib
---

# Summary

Imaging techniques are an important and growing source of scientific data, but the many different standards and file formats for representing and storing such data make streamlined analysis and data sharing cumbersome. To address this issue, the imaging community has recently developed an open and flexible file format specification (OME-NGFF) [@Moore2021-dw] and a concrete implementation using the Zarr file format (OME-Zarr) [@Moore2023-fv], which is increasingly adopted as the way to store and share multidimensional bioimaging data.

The aim of `ez-zarr` (pronounced “easy zarr”) is to provide high-level, read-only programmatic access to OME-Zarr files, abstracting away the complex, nested representation of these on disk. Internally, the OME-Zarr files are represented using a custom python class, which standardises simple operations such as metadata extraction, image subsetting and visualisation. In addition, the underlying image data, or subsets thereof, can be extracted as `numpy` arrays [@harris2020-np], which makes `ez-zarr` a valuable building block in more complex image analysis workflows that make use of tools that may not yet support OME-Zarrs. 

`ez-zarr` also facilitates working with collections of OME-Zarr files, such as images from high-content screens where each microwell in a plate is represented by one OME-Zarr file and the OME-Zarr plate collection provides necessary metadata. Such plate-based high-content screening experiments are becoming increasingly popular in research, and entire analysis workflows, like Fractal [@Fractal], now streamline the generation and processing of OME-Zarr datasets.

Finally, while `ez-zarr` is implemented in python and installable using `pip` and `conda`, we also provide an R wrapper package called `ezzarr` that enables the use of the python functions from R (https://github.com/fmicompbio/ezzarr).


# Statement of need

OME-Zarr is a relatively new file format and not yet supported by many image analysis tools. There are tools that allow accessing OME-Zarr files, amongst them interactive viewers, such as `napari` [@Sofroniew2024-cz] and `MoBIE` [@Pape2023-hr], and low level libraries to read and write OME-Zarrs, including `ome-zarr` [@ome-zarr], `ngio` [@ngio] and `zarr` [@zarr].

While these tools allow interactive exploration of images in OME-Zarr format, or getting low-level access to the contained data, we were not aware of tools that provide higher-level access and scriptable visualisation capabilities. Some more specialized projects also cover such tasks, but focus on specific types of imaging data, for example spatial omics data in the `SpatialData` framework [@Marconato2024-ie]. Furthermore, most available tools are implemented in python and do not have an equivalent interface in other languages like R. `ez-zarr` was designed to fill these gaps, building on the low-level file access functionality from the `zarr` library, and provide a minimal, abstract interface to OME-Zarr images or collections of such images from both python and R.

`ez-zarr` has been used productively by several research groups and in published research [@Schwayer2025]. 


# Examples

A typical use-case for `ez-zarr` is the scripted exploration of microscopy data at different resolutions, for example to perform quality control and to visualise the underlying imaging data for specific segmented objects. Figure 1 shows three example images, each created with a single `ez-zarr` function call.

![Three different plots with scale bars created from the same image: The full image with segmentation labels on the left, a zoomed-in version showing only a few objects in the middle, and a single object with different stainings shown as a false-colour image on the right. \label{fig:1}](figures/Figure1.png)

For collections of images, for example obtained from high-content microscopy experiments, `ez-zarr` provides a special `ImageList` container that streamlines import and visualisation of entire microwell plates (Figure 2). The `ImageList` class also implements subsetting and parallelisation over its items. 

![Example of a 24-well plate, in which only 15 of the wells were used in the experiment. The underlying data (a folder containing 15 OME-Zarrs) can be directly imported and visualised using `ez-zarr`. Here, we have additionally overlaid the intensity images with their corresponding segmentation labels, also stored in the OME-Zarr. \label{fig:2}](figures/Figure2.png)


# Acknowledgements

We would like to thank Joel Lüthi, Jan Eglinger, Tim-Oliver Buchholz and the members of the Liberali lab for fruitful discussions, testing and feedback on the software. 

# References

