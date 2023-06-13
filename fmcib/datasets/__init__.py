import pandas as pd
import wget

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
