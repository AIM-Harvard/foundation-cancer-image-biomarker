from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm.contrib.concurrent import process_map

np.bool = bool


def process_malignancy_label(df):
    df = df[df["malignancy"] != 3]
    df["malignancy"] = df["malignancy"].map({1: 0, 2: 0, 4: 1, 5: 1})
    return df


def process_row(row, data_dir=None):
    _, row = row
    row["image_path"] = str(data_dir / (row["scan"] + ".mhd"))
    img = sitk.ReadImage(row["image_path"])

    centroid = []
    for p in row["centroid"].strip(" [ ] ").split(","):
        try:
            centroid.append(int(float(p)))
        except:
            pass

    centroid = [centroid[1], centroid[0], centroid[2]]
    centroid = img.TransformContinuousIndexToPhysicalPoint(centroid)

    row["coordX"] = centroid[0]
    row["coordY"] = centroid[1]
    row["coordZ"] = centroid[2]

    return row


def main(args):
    annotations_df = pd.read_csv(args.annotations_csv)

    rows = list(annotations_df.iterrows())
    results = process_map(partial(process_row, data_dir=args.data_dir), rows, max_workers=4)

    updated_annotations = pd.DataFrame(results)
    updated_annotations = process_malignancy_label(updated_annotations)

    # Limit columns of interest
    updated_annotations = updated_annotations[["malignancy", "coordX", "coordY", "coordZ", "image_path"]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    updated_annotations.to_csv(args.output_dir / "luna16_training_annotations.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "annotations_csv",
        help="Path to csv file with deeplesion annotations",
        type=Path,
    )
    parser.add_argument("data_dir", help="Path to directory where data is present", type=Path)
    parser.add_argument(
        "--output_dir",
        help="Path to directory where annotations will be stored",
        type=Path,
    )

    args = parser.parse_args()

    main(args)
