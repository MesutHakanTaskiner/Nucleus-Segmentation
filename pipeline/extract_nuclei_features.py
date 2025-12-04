import numpy as np
import pandas as pd
from skimage import measure


def extract_nuclei_features(label_map: np.ndarray, image_gray: np.ndarray) -> pd.DataFrame:
    props = measure.regionprops_table(
        label_map,
        intensity_image=image_gray,
        properties=(
            "label",
            "area",
            "perimeter",
            "centroid",
            "major_axis_length",
            "minor_axis_length",
            "eccentricity",
        ),
    )
    df = pd.DataFrame(props)
    if not df.empty:
        df["circularity"] = (4 * np.pi * df["area"]) / (df["perimeter"] ** 2 + 1e-6)
    return df
