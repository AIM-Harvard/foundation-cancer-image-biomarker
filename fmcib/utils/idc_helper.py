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

def download_from_manifest(df, save_dir, samples):
    # Instantiates a client
    storage_client = storage.Client()
    bucket = storage_client.bucket("idc-open-cr")
    logger.info("Downloading DICOM data from IDC (Imaging Data Commons) ...")
    (save_dir / "dicom").mkdir(exist_ok=True, parents=True)

    if samples is not None:
        assert "PatientID" in df.columns
        unique_elements = df['PatientID'].unique()


        selected_elements = np.random.choice(unique_elements, min(len(unique_elements), samples), replace=False)
        df = df[df['PatientID'].isin(selected_elements)]

    def download_file(row):
        fn = f'{row["gcs_url"].split("/")[-2]}/{row["gcs_url"].split("/")[-1]}'
        blob = bucket.blob(fn)

        current_save_dir = save_dir / "dicom" / row["PatientID"] / row["StudyInstanceUID"]
        current_save_dir.mkdir(exist_ok=True, parents=True)
        blob.download_to_filename(str(current_save_dir / f'{row["Modality"]}_{row["SeriesInstanceUID"]}_{row["InstanceNumber"]}.dcm'))


    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for idx, row in df.iterrows():
            futures.append(executor.submit(download_file, row))
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass


def download_LUNG1(path, samples=None):
    save_dir = Path(path).resolve()
    save_dir.mkdir(exist_ok=True, parents=True)

    logger.info("Downloading LUNG1 manifest from Dropbox ...")
    # Download LUNG1 data manifest, this is precomputed but any set of GCS dicom files can be used here
    wget.download(
        "https://www.dropbox.com/s/lkvv33nmepecyu5/nsclc_radiomics.csv?dl=1",
        out=f"{save_dir}/nsclc_radiomics.csv",
    )

    df = pd.read_csv(f"{save_dir}/nsclc_radiomics.csv")

    download_from_manifest(df, save_dir, samples)


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
