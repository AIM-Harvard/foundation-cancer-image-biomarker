import numpy as np
import pylidc as pl
from pylidc.utils import consensus
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict
import ipywidgets as widgets
from multiprocessing import Pool
from tqdm import tqdm
import sys, os

np.int = np.int32
np.bool = bool

# Attributes of interest among the nodules
attribute_keys = [
    "calcification",
    "internalStructure",
    "lobulation",
    "malignancy",
    "margin",
    "sphericity",
    "spiculation",
    "subtlety",
    "texture",
]
luna_annotations = pd.read_csv(
    "https://zenodo.org/record/3723295/files/annotations.csv"
)

random_choices = 0


def get_consensus_attribute(v):
    """
    Get the mean of v
    """
    return int(np.rint(np.mean(v)))


def get_consensus_bbox(anns):
    # Perform a consensus consolidation and 50% agreement level.
    # We pad the slices to add context for viewing.
    _, cbbox, _ = consensus(anns, clevel=0.5, pad=False)

    centroid = [el.start + int(0.5 * (el.stop - el.start)) for el in cbbox]
    return {"bbox": cbbox, "centroid": centroid}


nodule_attributes_dict = []

# Get list of series_uids. Construct a set of the values
# since duplicates exist and pylidc won't account for them
series_uids = set(luna_annotations["seriesuid"].values)
print(f"Total unique series: {len(series_uids)}")

for scan in tqdm(
    pl.query(pl.Scan).filter(pl.Scan.series_instance_uid.in_(series_uids)),
    total=len(series_uids),
):
    annotations = scan.cluster_annotations()
    annotations_curated = [anns for anns in annotations if len(anns) >= 3]

    for annotation_set in annotations_curated:
        row = defaultdict(list)
        anns = []
        for annotation in annotation_set:
            if annotation.diameter >= 3:
                anns.append(annotation)
                for key in attribute_keys:
                    row[key].append(getattr(annotation, key))

        # Get mode of attributes in the annotation set
        row = {
            k: get_consensus_attribute(v) for k, v in row.items() if k in attribute_keys
        }

        consensus_bbox = get_consensus_bbox(anns)

        row["bbox"] = consensus_bbox["bbox"]
        row["centroid"] = consensus_bbox["centroid"]
        row["scan"] = scan.series_instance_uid

        nodule_attributes_dict.append(row)


annotations_df = pd.DataFrame(nodule_attributes_dict)
folder = sys.argv[1]
if not os.path.exists(folder):
    os.mkdir(folder)
annotations_df.to_csv(f"{folder}/luna16_annotations.csv")
