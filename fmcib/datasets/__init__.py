import os
import random
from pathlib import Path

import monai
import numpy as np
import pandas as pd
import SimpleITK as sitk
import wget
from loguru import logger

from .ssl_radiomics_dataset import SSLRadiomicsDataset


def get_lung1_clinical_data():
    wget.download(
        "https://www.dropbox.com/s/ulp8t21eunep21y/NSCLC%20Radiomics%20Lung1.clinical-version3-Oct%202019.csv?dl=1",
        out="/tmp/lung1_clinical.csv",
    )
    return pd.read_csv("/tmp/lung1_clinical.csv")


def get_radio_clinical_data():
    wget.download(
        "https://www.dropbox.com/s/mtpynjof550ulfo/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv?dl=1",
        out=f"/tmp/radio_clinical.csv",
    )
    return pd.read_csv("/tmp/radio_clinical.csv")


def get_lung1_foundation_features():
    wget.download(
        "https://www.dropbox.com/s/ypbb2iogq3bsq5v/lung1.csv?dl=1",
        out=f"/tmp/lung1_foundation_features.csv",
    )
    df = pd.read_csv("/tmp/lung1_foundation_features.csv")
    filtered_df = df.filter(like="pred")
    filtered_df = filtered_df.reset_index()  # reset the index
    filtered_df["PatientID"] = df["PatientID"]
    return filtered_df


def get_radio_foundation_features():
    wget.download(
        "https://www.dropbox.com/s/pwl4rdlvp9jirar/radio.csv?dl=1",
        out=f"/tmp/radio_foundation_features.csv",
    )

    df = pd.read_csv("/tmp/radio_foundation_features.csv")
    filtered_df = df.filter(like="pred")
    filtered_df = filtered_df.reset_index()  # reset the index
    filtered_df["PatientID"] = df["Case ID"]
    return filtered_df


def generate_dummy_data(dir_path, size=10):
    path = Path(dir_path).resolve()
    path.mkdir(exist_ok=True, parents=True)

    row_list = []
    for i in range(size):
        row = create_dummy_row((32, 128, 128), str(path / f"dummy_{i}.nii.gz"))
        row_list.append(row)

    df = pd.DataFrame(row_list)
    df.to_csv(path / "dummy.csv", index=False)

    logger.info(f"Generated dummy data at {path}/dummy.csv")


def create_dummy_row(size, output_filename):
    """
    Function to create a dummy row with path to an image and seed point corresponding to the image
    """

    # Create a np array initialized with random values between -1024 and 2048
    np_image = np.random.randint(-1024, 2048, size, dtype=np.int16)

    # Create an itk image from the numpy array
    itk_image = sitk.GetImageFromArray(np_image)

    # Save itk image to file with the given output filename
    sitk.WriteImage(itk_image, output_filename)

    x, y, z = generate_random_seed_point(itk_image.GetSize())

    # Convert to global coordinates
    x, y, z = itk_image.TransformContinuousIndexToPhysicalPoint((x, y, z))

    return {
        "image_path": output_filename,
        "PatientID": random.randint(0, 100000),
        "coordX": x,
        "coordY": y,
        "coordZ": z,
        "label": random.randint(0, 1),
    }


def generate_random_seed_point(image_size):
    """
    Function to generate a random x, y, z coordinate within the image
    """
    x = random.randint(0, image_size[0] - 1)
    y = random.randint(0, image_size[1] - 1)
    z = random.randint(0, image_size[2] - 1)

    return (x, y, z)
