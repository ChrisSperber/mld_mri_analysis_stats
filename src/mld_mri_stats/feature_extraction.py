"""Functions for the extraction of imaging features."""

import warnings

import numpy as np


def extract_roi_quantile(
    image: np.ndarray,
    mask: np.ndarray,
    quantile: float = 0.5,
) -> float:
    """Extract a quantile of image values within a binary 3D ROI mask.

    Image and mask must have identical shapes. The mask must be binary
    and non-empty. Quantile must be between 0 and 1 (default: 0.5 [median]).

    Non-finite image values within the ROI are excluded with a warning.
    Returns NaN if no finite values remain.
    """
    if image.ndim != 3 or mask.ndim != 3:  # noqa: PLR2004
        raise ValueError("Image and mask must be three-dimensional.")

    if image.shape != mask.shape:
        raise ValueError(
            f"Image and mask shapes differ: {image.shape} vs. {mask.shape}."
        )

    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("ROI mask contains values other than 0 and 1.")

    if not np.any(mask == 1):
        raise ValueError("ROI mask is empty.")

    if not 0 <= quantile <= 1:
        raise ValueError("Quantile must be between 0 and 1.")

    values = image[mask == 1]

    finite = np.isfinite(values)
    if not np.all(finite):
        warnings.warn(
            f"Excluded {np.sum(~finite)} of {values.size} non-finite ROI values.",
            RuntimeWarning,
            stacklevel=2,
        )
        values = values[finite]

    if values.size == 0:
        return np.nan

    return float(np.quantile(values, quantile))
