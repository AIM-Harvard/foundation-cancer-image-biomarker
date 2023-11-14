from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import scipy
from joblib import Parallel, delayed
from scipy.stats import bootstrap, permutation_test
from tqdm import tqdm

pio.templates["custom"] = go.layout.Template(
    layout=go.Layout(
        colorway=px.colors.qualitative.D3,
    )
)
from utils import get_model_comparison_stats, get_model_stats

path = Path("../outputs/predictions/task1")
implementation_dict = {
    "Our modified SimCLR": [csv_path for csv_path in path.glob("foundation_features*.csv")],
    "SimCLR": [csv_path for csv_path in path.glob("simclr_features*.csv")],
    "SwAV": [csv_path for csv_path in path.glob("swav_features*.csv")],
    "NNCLR": [csv_path for csv_path in path.glob("nnclr_features*.csv")],
    "Auto-Encoder": [csv_path for csv_path in path.glob("autoencoder_features*.csv")],
}

implementation_rank = {key: i for i, key in enumerate(implementation_dict.keys())}
pbar = tqdm(total=len(implementation_dict) * len(implementation_dict["SimCLR"]))
results = []

N_RESAMPLES = 1000
for implementation_name, implementation_list in implementation_dict.items():
    for model_prediction_csv in implementation_list:
        data_percentage = (
            float(model_prediction_csv.stem.split("_")[-2]) / 100 if len(model_prediction_csv.stem.split("_")) > 2 else 1.0
        )
        df = pd.read_csv(model_prediction_csv)

        for i in range(8):
            if f"conf_scores_class_{i}" not in df.columns:
                df[f"conf_scores_class_{i}"] = 0

        pred_set = (df["Coarse_lesion_type"].values, df.filter(like="conf_scores").values)
        map_ci = get_model_stats(
            *pred_set,
            fn="mean_average_precision",
            nsamples=N_RESAMPLES,
        )

        ba_ci = get_model_stats(
            *pred_set,
            fn="balanced_accuracy",
            nsamples=N_RESAMPLES,
        )

        row = {
            "Implementation": implementation_name,
            "Data Percentage": data_percentage,
            "mAP": map_ci[0],
            "mAP_low_CI": map_ci[1][0],
            "mAP_high_CI": map_ci[1][1],
            "BA": ba_ci[0],
            "BA_low_CI": ba_ci[1][0],
            "BA_high_CI": ba_ci[1][1],
        }

        # Compute statistics for comparison between this implementation and all other ones (difference CI and p-value)
        compare_impementations = {k: v for k, v in implementation_dict.items() if k != implementation_name}
        for _implementation_name, _implementations_list in compare_impementations.items():
            for _model_prediction_csv in _implementations_list:
                _data_percentage = (
                    float(_model_prediction_csv.stem.split("_")[-2]) / 100
                    if len(_model_prediction_csv.stem.split("_")) > 2
                    else 1.0
                )
                if data_percentage == _data_percentage:
                    _df = pd.read_csv(_model_prediction_csv)
                    # Check if 8 columns with conf_scores_class_{idx} exist, if not add a column with zeros for missing
                    for i in range(8):
                        if f"conf_scores_class_{i}" not in _df.columns:
                            _df[f"conf_scores_class_{i}"] = 0

                    _pred = _df.filter(like="conf_scores").values
                    _pred_set = (*pred_set, _pred)

                    perm_test = get_model_comparison_stats(
                        *_pred_set,
                        fn="balanced_accuracy",
                        nsamples=N_RESAMPLES,
                    )

                    row[f"BA_diff_CI_low_{_implementation_name}"] = perm_test[0][0]
                    row[f"BA_diff_CI_high_{_implementation_name}"] = perm_test[0][1]
                    row[f"BA_pval_{_implementation_name}"] = perm_test[1]

                    perm_test = get_model_comparison_stats(
                        *_pred_set,
                        fn="mean_average_precision",
                        nsamples=N_RESAMPLES,
                    )

                    row[f"mAP_diff_CI_low_{_implementation_name}"] = perm_test[0][0]
                    row[f"mAP_diff_CI_high_{_implementation_name}"] = perm_test[0][1]
                    row[f"mAP_pval_{_implementation_name}"] = perm_test[1]

        results.append(row)
        pbar.update(1)

results_df = pd.DataFrame(results)
results_df["Implementation_Rank"] = results_df["Implementation"].map(implementation_rank)
results_df.sort_values(by=["Data Percentage", "Implementation_Rank"], inplace=True, ascending=True)
results_df.drop("Implementation_Rank", axis=1, inplace=True)

results_df.to_csv("result_csvs/pretraining_comparison.csv", index=False)
