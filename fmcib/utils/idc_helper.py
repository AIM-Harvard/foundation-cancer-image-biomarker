import concurrent.futures
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import pydicom_seg
import SimpleITK as sitk
import wget
from dcmrtstruct2nii import dcmrtstruct2nii
from dcmrtstruct2nii.adapters.input.image.dcminputadapter import DcmInputAdapter
from dcmrtstruct2nii.adapters.output.niioutputadapter import NiiOutputAdapter
from google.cloud import storage
from loguru import logger
from tqdm import tqdm


def dcmseg2nii(dcmseg_path, output_dir, tag=""):
    dcm = pydicom.dcmread(dcmseg_path)
    reader = pydicom_seg.SegmentReader()
    result = reader.read(dcm)

    for segment_number in result.available_segments:
        image = result.segment_image(segment_number)  # lazy construction
        sitk.WriteImage(image, output_dir + f"/{tag}{segment_number}.nii.gz", True)


def download_from_manifest(df, save_dir, samples):
    # Instantiates a client
    storage_client = storage.Client()
    logger.info("Downloading DICOM data from IDC (Imaging Data Commons) ...")
    (save_dir / "dicom").mkdir(exist_ok=True, parents=True)

    if samples is not None:
        assert "PatientID" in df.columns
        rows_with_annotations = df[df["Modality"].isin(["RTSTRUCT", "SEG"])]
        unique_elements = rows_with_annotations["PatientID"].unique()
        selected_elements = np.random.choice(unique_elements, min(len(unique_elements), samples), replace=False)
        df = df[df["PatientID"].isin(selected_elements)]

    def download_file(row):
        bucket_name, directory, file = row["gcs_url"].split("/")[-3:]
        fn = f"{directory}/{file}"
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(fn)

        current_save_dir = save_dir / "dicom" / row["PatientID"] / row["StudyInstanceUID"]
        current_save_dir.mkdir(exist_ok=True, parents=True)
        blob.download_to_filename(
            str(current_save_dir / f'{row["Modality"]}_{row["SeriesInstanceUID"]}_{row["InstanceNumber"]}.dcm')
        )

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


def download_RADIO(path, samples=None):
    save_dir = Path(path).resolve()
    save_dir.mkdir(exist_ok=True, parents=True)

    logger.info("Downloading RADIO manifest from Dropbox ...")
    # Download RADIO data manifest, this is precomputed but any set of GCS dicom files can be used here
    wget.download(
        "https://www.dropbox.com/s/nhh1tb0rclrb7mw/nsclc_radiogenomics.csv?dl=1",
        out=f"{save_dir}/nsclc_radiogenomics.csv",
    )

    df = pd.read_csv(f"{save_dir}/nsclc_radiogenomics.csv")

    download_from_manifest(df, save_dir, samples)


def process_series_dir(series_dir):
    # Check if RTSTRUCT file exists
    rtstuct_files = list(series_dir.glob("*RTSTRUCT*"))
    seg_files = list(series_dir.glob("*SEG*"))

    if len(rtstuct_files) != 0:
        dcmrtstruct2nii(str(rtstuct_files[0]), str(series_dir), str(series_dir))

    elif len(seg_files) != 0:
        dcmseg2nii(str(seg_files[0]), str(series_dir), tag="GTV-")

        series_id = str(list(series_dir.glob("CT*.dcm"))[0]).split("_")[-2]
        dicom_image = DcmInputAdapter().ingest(str(series_dir), series_id=series_id)
        nii_output_adapter = NiiOutputAdapter()
        nii_output_adapter.write(dicom_image, f"{series_dir}/image", gzip=True)
    else:
        logger.warning("Skipped file without any RTSTRUCT or SEG file")
        return None

    image = sitk.ReadImage(str(series_dir / "image.nii.gz"))
    mask = sitk.ReadImage(str(list(series_dir.glob("*GTV-1*"))[0]))

    # Get centroid from label shape filter
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(mask)

    try:
        centroid = label_shape_filter.GetCentroid(255)
    except:
        centroid = label_shape_filter.GetCentroid(1)

    x, y, z = centroid

    row = {
        "image_path": str(series_dir / "image.nii.gz"),
        "PatientID": series_dir.parent.name,
        "coordX": x,
        "coordY": y,
        "coordZ": z,
    }

    return row


def build_image_seed_dict(path, samples=None):
    sorted_dir = Path(path).resolve()
    series_dirs = [x.parent for x in sorted_dir.rglob("*.dcm")]
    series_dirs = sorted(list(set(series_dirs)))

    logger.info("Converting DICOM files to NIFTI ...")

    if samples is None:
        samples = len(series_dirs)

    rows = []

    num_workers = os.cpu_count()  # Adjust this value based on the number of available CPU cores
    with concurrent.futures.ProcessPoolExecutor(num_workers) as executor:
        processed_rows = list(tqdm(executor.map(process_series_dir, series_dirs[:samples]), total=samples))

    rows = [row for row in processed_rows if row]
    return pd.DataFrame(rows)
