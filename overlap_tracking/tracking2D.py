from typing import Iterable

import numpy as np
import polars as pl
from numpy.typing import NDArray


def single_timestep_overlap_tracking(
    im1: NDArray[np.int32 | np.int64],
    im2: NDArray[np.int32 | np.int64],
    ignore_labels: Iterable[int],
) -> dict[int, int]:
    """
    Function for tracking objects in an image sequence based on maximum area
    overlap.

    Args:
        im1 (NDArray): image at time t
        im2 (NDArray): image at time t+1
        ignore_labels (Iterable[int]): ignore these labels when recording
            overlap areas, e.g. background label.
    Returns:
        (dict[int, int]): dictionary mapping object labels in im1 to
            corresponding object labels in im2.
    """

    # get a list of unique labels in current image, removing labels in 'ignore_labels'
    current_labels: NDArray = np.setdiff1d(np.unique(im1), ignore_labels)

    overlap_dict: list[dict] = []

    # we iterate over each label and add record all overalp areas
    for label in current_labels:
        other_values: np.array = im2[im1 == label]

        values, counts = np.unique(other_values, return_counts=True)

        for index, value in enumerate(values):
            if value not in ignore_labels:
                overlap_dict.append(
                    {
                        "current_label": label,
                        "next_label": value,
                        "overlap_count": counts[index],
                    }
                )

    labelmap = {}

    overlap_df: pl.DataFrame = pl.DataFrame(overlap_dict)
    overlap_df = overlap_df.sort(by="overlap_count", descending=True)

    for row in overlap_df.iter_rows():
        if (not row[0] in labelmap) and (not row[1] in labelmap.values()):
            labelmap[row[0]] = row[1]

    return labelmap
