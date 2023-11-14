from scipy.stats import bootstrap, permutation_test
import pandas as pd
from pathlib import Path
from functools import partial
from tqdm import tqdm
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates["custom"] = go.layout.Template(
    layout=go.Layout(
        colorway=px.colors.qualitative.D3,
    )
)

from utils import get_model_stats, get_model_comparison_stats

path = Path("../outputs/predictions/task2")

implementation_dict = {
    "Foundation (Features)": [csv_path for csv_path in path.glob("foundation_features*.csv")],
    "Foundation (Finetuned)": [csv_path for csv_path in path.glob("foundation_finetuned*.csv")],
    "Supervised": [csv_path for csv_path in path.glob("supervised_random*.csv")],
    "Supervised (Finetuned)": [csv_path for csv_path in path.glob("supervised_finetuned*.csv")],
    "Med3D (Features)": [csv_path for csv_path in path.glob("med3d_features*.csv")], 
    "Models Genesis (Features)": [csv_path for csv_path in path.glob("modelsgen_features*.csv")], 
    "Med3D (Finetuned)": [csv_path for csv_path in path.glob("med3d_finetuned*.csv")], 
    "Models Genesis (Finetuned)": [csv_path for csv_path in path.glob("modelsgen_finetuned*.csv")], 
}

implementation_rank = {key: i for i, key in enumerate(implementation_dict.keys())}

pbar = tqdm(total=len(implementation_dict) * len(implementation_dict["Supervised"]))
results = []

# We use 1000 resamples in the study, but for the sake of time we use reproduce results with 10 here
N_RESAMPLES = 1000

for implementation_name, implementation_list in implementation_dict.items():
    for model_prediction_csv in implementation_list:
        data_percentage = (
            float(model_prediction_csv.stem.split("_")[-2]) / 100 if len(model_prediction_csv.stem.split("_")) > 2 else 1.0
        )
        df = pd.read_csv(model_prediction_csv)
        pred_set = (df["malignancy"].values, df["conf_scores_class_1"].values if "conf_scores_class_1" in df.columns else df["conf_scores_class"].values)

        map_ci = get_model_stats(
            *pred_set,
            fn="average_precision_score",
            nsamples=N_RESAMPLES,
        )

        auc_ci = get_model_stats(
            *pred_set,
            fn="roc_auc_score",
            nsamples=N_RESAMPLES,
        )
        row = {
            "Implementation": implementation_name,
            "Data Percentage": data_percentage,
            "mAP": map_ci[0],
            "mAP_low_CI": map_ci[1][0],
            "mAP_high_CI": map_ci[1][1],
            "AUC": auc_ci[0],
            "AUC_low_CI": auc_ci[1][0],
            "AUC_high_CI": auc_ci[1][1],
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
                    _pred = _df["conf_scores_class_1"].values if "conf_scores_class_1" in _df.columns else _df["conf_scores_class"].values
                    _pred_set = (*pred_set, _pred)

                    perm_test = get_model_comparison_stats(
                        *_pred_set,
                        fn="roc_auc_score",
                        nsamples=N_RESAMPLES,
                    )

                    row[f"AUC_diff_CI_low_{_implementation_name}"] = perm_test[0][0]
                    row[f"AUC_diff_CI_high_{_implementation_name}"] = perm_test[0][1]
                    row[f"AUC_pval_{_implementation_name}"] = perm_test[1]


                    perm_test = get_model_comparison_stats(
                        *_pred_set,
                        fn="average_precision_score",
                        nsamples=N_RESAMPLES,
                    )

                    row[f"mAP_diff_CI_low_{_implementation_name}"] = perm_test[0][0]
                    row[f"mAP_diff_CI_high_{_implementation_name}"] = perm_test[0][1]
                    row[f"mAP_pval_{_implementation_name}"] = perm_test[1]


        results.append(row)
        pbar.update(1)
results_df = pd.DataFrame(results)
results_df['Implementation_Rank'] = results_df['Implementation'].map(implementation_rank)
results_df.sort_values(by=["Data Percentage", "Implementation_Rank"], inplace=True, ascending=True)
results_df.drop('Implementation_Rank', axis=1, inplace=True)
results_df.to_csv("result_csvs/task2.csv")