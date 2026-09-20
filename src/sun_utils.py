import cv2
import numpy as np


def rotate_for_sun(image, sun_azimuth_angle):
    """
    Rotate a lunar image according to the competition specification.

    Competition requirement:
        Rotate the image counter-clockwise by
        -sun_azimuth_angle degrees.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image, expected shape (H, W).

    sun_azimuth_angle : float
        Sun azimuth angle in degrees.

    Returns
    -------
    np.ndarray
        Rotated image with the same dimensions as the input.

    Coordinate convention
    ---------------------
    Image coordinates:
        x increases to the right
        y increases downward

    OpenCV:
        positive rotation parameter = counter-clockwise
        visual rotation

    Therefore:
        rotation_angle = -sun_azimuth_angle
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")

    if image.ndim != 2:
        raise ValueError(
            f"Expected grayscale image with shape (H, W), "
            f"got shape {image.shape}"
        )

    if not np.isfinite(sun_azimuth_angle):
        raise ValueError(
            "sun_azimuth_angle must be finite"
        )

    height, width = image.shape

    center = (
        width / 2.0,
        height / 2.0
    )

    rotation_angle = -float(sun_azimuth_angle)

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        rotation_angle,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    return rotated