import os
import subprocess

import pandas as pd

from fmcib.datasets import generate_dummy_data
from fmcib.models import fmcib_model
from fmcib.run import get_features


def test_dummy_features():
    """
    Test feature extraction on dummy data.

    Generates a small dummy dataset, extracts features using the model,
    and verifies that the expected number of features (4096) are produced.
    """
    generate_dummy_data("data", size=10)
    df = pd.read_csv("data/dummy.csv")
    feature_df = get_features("data/dummy.csv")

    assert feature_df.filter(like="pred").shape[1] == 4096


def test_dummy_finetuning():
    """
    Test model finetuning workflow on dummy data.

    Generates a larger dummy dataset, ensures model weights exist by downloading
    if needed, and runs a training job using the test configuration. Verifies
    that the training process completes successfully.
    """
    generate_dummy_data("data", size=50)
    if not os.path.exists("./model_weights.torch"):
        fmcib_model()

    subprocess.run(["lighter", "fit", "--config_file", "./tests/integration/config/test.yaml"], check=True)
