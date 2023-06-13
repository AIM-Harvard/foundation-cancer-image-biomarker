import sys
from functools import partial

import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


def process_row(row):
    _, row = row
    img = sitk.ReadImage(row["image_path"])
    physical_coordinates = img.TransformIndexToPhysicalPoint([int(row[f"centroid_{t}"]) for t in ["x", "y", "z"]])
    row["coordX"] = physical_coordinates[0]
    row["coordY"] = physical_coordinates[1]
    row["coordZ"] = physical_coordinates[2]
    return row


deeplesion_annotations = pd.read_csv("./annotations/deeplesion_annotations.csv")
deeplesion_annotations["image_path"] = deeplesion_annotations["Volume_fn"].apply(
    lambda x: f"{sys.argv[1]}/DeepLesion/Images_nifti/{x}"
)

rows = list(deeplesion_annotations.iterrows())
results = process_map(process_row, rows, max_workers=6)

updated_annotations = pd.DataFrame(results)
updated_annotations.to_csv("./annotations/deeplesion_annotations_training.csv", index=False)
