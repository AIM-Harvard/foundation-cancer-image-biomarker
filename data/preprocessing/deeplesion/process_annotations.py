from pathlib import Path
from unittest import result
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString

# Compute the point where two line segments intersect
def get_intersection_point(line_segment1, line_segment2):
    p1 = Point(line_segment1[0], line_segment1[1])
    p2 = Point(line_segment1[2], line_segment1[3])
    p3 = Point(line_segment2[0], line_segment2[1])
    p4 = Point(line_segment2[2], line_segment2[3])

    line1 = LineString([p1, p2])
    line2 = LineString([p3, p4])

    intersection = line1.intersection(line2)

    if intersection.type == "LineString":
        return None

    return intersection.x, intersection.y


def get_nodule_dimensions(row):
    spacings = [float(x) for x in row["Spacing_mm_px_"].split(",")]
    lesion_diameters_px = [float(x) for x in row["Lesion_diameters_Pixel_"].split(",")]
    recist_diameters = [float(x) for x in row["Measurement_coordinates"].split(",")]
    bbox = [int(float(x)) for x in row["Bounding_boxes"].split(",")]

    lesion_diameters = [
        diameter_px * spacing
        for diameter_px, spacing in zip(lesion_diameters_px, spacings[:-1])
    ]

    # MAJOR: Assume that the lesion's z diameter is the largest of x-y diameters
    lesion_diameters.append(np.max(lesion_diameters))

    # Compute centroid of the lesion using the RECIST diameters
    long_axis = [*recist_diameters[:4]]
    short_axis = [*recist_diameters[4:]]

    intersection_point = get_intersection_point(long_axis, short_axis)

    if intersection_point is None:
        return 0, lesion_diameters, bbox, spacings

    centroid = (
        *intersection_point,
        int(row["Slice_range"].split(",")[1]) - int(row["Key_slice_index"]) + 1,
    )

    return centroid, lesion_diameters, bbox, spacings


def main(args):
    annotations_df = pd.read_csv(args.annotations_csv)
    annotations_df["Volume_fn"] = (
        annotations_df["File_name"].apply(lambda x: x.rsplit("_", 1)[0])
        + "_"
        + annotations_df["Slice_range"].apply(
            lambda x: "{:03d}-{:03d}".format(*[int(i) for i in x.split(",")])
        )
        + ".nii.gz"
    )

    processed_df = annotations_df[
        [
            "Volume_fn",
            "Train_Val_Test",
            "DICOM_windows",
            "Patient_gender",
            "Patient_age",
            "Coarse_lesion_type",
            "Possibly_noisy",
        ]
    ]

    processed_df[
        ["centroid", "lesion_diameters", "bbox", "spacing"]
    ] = annotations_df.apply(
        lambda row: get_nodule_dimensions(row), axis=1, result_type="expand"
    )

    # Remove rows where intersection point is not found (i.e. invalid annotations)
    processed_df = processed_df[processed_df["centroid"] != 0]

    expand_rows = ["centroid", "lesion_diameters", "spacing"]

    for row in expand_rows:
        processed_df[[f"{row}_{d}" for d in ["x", "y", "z"]]] = processed_df[row].apply(
            pd.Series
        )

    processed_df = processed_df.drop(expand_rows, axis=1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(args.output_dir / "deeplesion_annotations.csv")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "annotations_csv",
        help="Path to csv file with deeplesion annotations",
        type=Path,
    )
    parser.add_argument(
        "output_dir",
        help="Path to directory where processed annotations will be stored",
        type=Path,
    )
    args = parser.parse_args()

    main(args)
