"""
PET Image Harmonization

Two harmonization approaches are implemented:

H1:
    Standardize the physical field of view to 500 x 500 mm
    by cropping or padding, followed by resizing to 125 x 125.

H2:
    Keep the original scanned area and directly resize
    each PET slice to 125 x 125.

Based on the preprocessing pipeline developed for the
PET prostate cancer classification project.
"""

import numpy as np
from scipy.ndimage import zoom


TARGET_SHAPE = (125, 125)
TARGET_FOV_MM = 500


def harmonize_h2(pet_slice, mask_slice):
    """
    H2 harmonization.

    Resize PET image and mask directly to 125 x 125
    without cropping or padding.

    Parameters
    ----------
    pet_slice : numpy.ndarray
        2D PET image.

    mask_slice : numpy.ndarray
        Corresponding 2D segmentation mask.

    Returns
    -------
    pet_h2 : numpy.ndarray
        Harmonized PET image.

    mask_h2 : numpy.ndarray
        Harmonized segmentation mask.
    """

    zooms = tuple(
        x / y
        for x, y in zip(
            TARGET_SHAPE,
            (pet_slice.shape[0], pet_slice.shape[1])
        )
    )

    # Linear interpolation for PET image
    pet_h2 = zoom(
        pet_slice,
        zooms,
        order=1
    )

    # Nearest-neighbour interpolation for mask
    mask_h2 = zoom(
        mask_slice,
        zooms,
        order=0
    )

    return pet_h2, mask_h2


def harmonize_h1(
    pet_slice,
    mask_slice,
    voxel_size
):
    """
    H1 harmonization.

    Crop or pad the PET image and mask so that their
    physical field of view is approximately 500 x 500 mm.

    The resulting images are then resized to 125 x 125,
    corresponding to approximately 4 x 4 mm pixels.

    Parameters
    ----------
    pet_slice : numpy.ndarray
        2D PET image.

    mask_slice : numpy.ndarray
        Corresponding segmentation mask.

    voxel_size : tuple
        PET voxel dimensions in millimetres.

    Returns
    -------
    pet_h1 : numpy.ndarray
        Harmonized PET image.

    mask_h1 : numpy.ndarray
        Harmonized segmentation mask.
    """

    # Work on copies
    pet_slice = pet_slice.copy()
    mask_slice = mask_slice.copy()

    dims = pet_slice.shape

    # Physical image dimensions in millimetres
    total_size = np.array([
        voxel_size[0] * dims[0],
        voxel_size[1] * dims[1]
    ])

    # Determine amount of cropping or padding
    change_0 = int(
        np.round(
            (total_size[0] - TARGET_FOV_MM)
            / (voxel_size[0] * 2)
        )
    )

    change_1 = int(
        np.round(
            (total_size[1] - TARGET_FOV_MM)
            / (voxel_size[1] * 2)
        )
    )

    # Use minimum PET intensity for padding
    pad_value = np.min(pet_slice)

    # --------------------------------------------------
    # First image dimension
    # --------------------------------------------------

    if change_0 > 0:
        pet_slice = pet_slice[
            change_0:(dims[0] - change_0),
            :
        ]

        mask_slice = mask_slice[
            change_0:(dims[0] - change_0),
            :
        ]

    elif change_0 < 0:

        change_0 = abs(change_0)

        pet_slice = np.pad(
            pet_slice,
            ((change_0, change_0), (0, 0)),
            "constant",
            constant_values=pad_value
        )

        mask_slice = np.pad(
            mask_slice,
            ((change_0, change_0), (0, 0)),
            "constant",
            constant_values=0
        )

    # --------------------------------------------------
    # Second image dimension
    # --------------------------------------------------

    if change_1 > 0:

        pet_slice = pet_slice[
            :,
            change_1:(pet_slice.shape[1] - change_1)
        ]

        mask_slice = mask_slice[
            :,
            change_1:(mask_slice.shape[1] - change_1)
        ]

    elif change_1 < 0:

        change_1 = abs(change_1)

        pet_slice = np.pad(
            pet_slice,
            ((0, 0), (change_1, change_1)),
            "constant",
            constant_values=pad_value
        )

        mask_slice = np.pad(
            mask_slice,
            ((0, 0), (change_1, change_1)),
            "constant",
            constant_values=0
        )

    # --------------------------------------------------
    # Resize to 125 x 125
    # --------------------------------------------------

    zooms = tuple(
        x / y
        for x, y in zip(
            TARGET_SHAPE,
            (
                pet_slice.shape[0],
                pet_slice.shape[1]
            )
        )
    )

    pet_h1 = zoom(
        pet_slice,
        zooms,
        order=1
    )

    mask_h1 = zoom(
        mask_slice,
        zooms,
        order=0
    )

    return pet_h1, mask_h1
