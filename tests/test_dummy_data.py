import pandas as pd

from fmcib.datasets import generate_dummy_data
from fmcib.run import get_features


def test_dummy_features():
    generate_dummy_data("data", size=10)
    df = pd.read_csv("data/dummy.csv")
    feature_df = get_features("data/dummy.csv")

    assert feature_df.filter(like="pred").shape[1] == 4096
