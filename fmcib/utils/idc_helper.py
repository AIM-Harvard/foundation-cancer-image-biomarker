import concurrent.futures
import subprocess
from pathlib import Path

import google.cloud.storage as gcs
import numpy as np
import pandas as pd
import SimpleITK as sitk
import wget
from dcmrtstruct2nii import dcmrtstruct2nii
from google.cloud import storage
from loguru import logger
from tqdm import tqdm

from .download_utils import bar_progress


def download_LUNG1(path):
    save_dir = Path(path).resolve()
    save_dir.mkdir(exist_ok=True, parents=True)

    logger.info("Downloading LUNG1 manifest from Dropbox ...")
    # Download LUNG1 data manifest, this is precomputed but any set of GCS dicom files can be used here
    wget.download(
        "https://www.dropbox.com/s/tbywsmxln5yatxw/gcs_lung1_paths.txt?dl=1",
        bar=bar_progress,
        out=f"{save_dir}/gcs_lung1_paths.txt",
    )

    # Instantiates a client
    storage_client = storage.Client()
    bucket = storage_client.bucket("idc-open-cr")

    logger.info("Downloading LUNG1 DICOM data from IDC (Imaging Data Commons) ...")

    (save_dir / "dicom").mkdir(exist_ok=True, parents=True)

    # The name of the file to download
    with open(f"{save_dir}/gcs_lung1_paths.txt", "r") as f:
        # Define a function to download a single file
        def download_file(fn):
            # Get the generation number of the blob
            blob = bucket.blob(fn.strip("\n"))

            # Generate the download URL with the generation number
            blob.download_to_filename(f"{str(save_dir)}/dicom/{fn.split('/')[-1]}")

        # Use a ThreadPoolExecutor to download multiple files in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for fn in f.readlines():
                futures.append(executor.submit(download_file, fn))
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                pass

    logger.info("Sorting files using dicomsort ...")

    # Sort the downloaded DICOM data
    # DICOM sort
    command = ["dicomsort"]
    command += [f"{save_dir}/dicom"]
    command += [f"{save_dir}/sorted/%PatientID/%StudyInstanceUID/%Modality_%SeriesInstanceUID_%InstanceNumber.dcm"]
    command += ["--keepGoing"]

    subprocess.run(command)


def build_image_seed_dict(path, samples=10):
    sorted_dir = Path(path).resolve()
    series_dirs = [x.parent for x in sorted_dir.rglob("*.dcm")]
    series_dirs = sorted(list(set(series_dirs)))

    logger.info("Converting DICOM files to NIFTI ...")

    rows = []
    for idx, series_dir in tqdm(enumerate(series_dirs), total=samples):
        if idx == samples:
            break

        dcmrtstruct2nii(str(list(series_dir.glob("*RTSTRUCT*"))[0]), str(series_dir), str(series_dir))

        image = sitk.ReadImage(str(series_dir / "image.nii.gz"))
        mask = sitk.ReadImage(str(list(series_dir.glob("*[gG][tT][vV]*"))[0]))

        print(np.unique(sitk.GetArrayFromImage(mask)))

        # Get centroid from label shape filter
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter.Execute(mask)
        centroid = label_shape_filter.GetCentroid(255)
        x, y, z = centroid

        row = {
            "image_path": str(series_dir / "image.nii.gz"),
            "PatientID": series_dir.parent.name,
            "coordX": x,
            "coordY": y,
            "coordZ": z,
        }

        rows.append(row)

    return pd.DataFrame(rows)
