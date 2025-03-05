# %% [markdown]
# ## Inter-reader and test-retest stability of foundation model against supervised methods corresponding to Extended Data Figure 3

import pickle

import lifelines
import matplotlib.pyplot as plt
import numpy as np

# %%
import pandas as pd
import pingouin as pg
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from scipy import stats
from scipy.special import softmax
from sklearn.metrics import r2_score, roc_auc_score

# %% [markdown]
# ## Inter-reader Stability

# %% [markdown]
# For the inter-reader variation evaluation, we used the LUNG1 dataset and generated 50 random perturbations sampled from a three-dimensional multivariate normal distribution with zero mean and diagonal covariance matrix for each seed point. Across each dimension, a variance of 16 voxels was used for generating samples. We generated predictions on perturbed seed points using the best foundation and supervised model,  resulting in 50 different prediction models for each

# %%
sup_df = pd.read_csv("../outputs/stability/supervised/lung1_preds.csv")
foundation_df = pd.read_csv("../outputs/stability/foundation_features/lung1_preds.csv")
original = pd.read_csv("../data/preprocessing/lung1/annotations/annotations.csv")


# %%
def compute_stability_metrics(df):
    trials = 50

    results_df = pd.DataFrame()

    for n_trial in range(trials):
        trial_df = df[df["trial"] == n_trial]

        # For the feature based model, load the model and predict
        pred = np.vstack([trial_df["pred_0"].values, trial_df["pred_1"].values])

        pred = softmax(pred, axis=0)
        auc = roc_auc_score(trial_df["survival"], pred[1])
        ci = lifelines.utils.concordance_index(trial_df["Survival.time"], pred[1], trial_df["deadstatus.event"])

        row = {
            "trial": n_trial,
            "AUC": auc,
            "CI": ci,
            "x": original["coordX"].values - trial_df["coordX"].values,
            "y": original["coordY"].values - trial_df["coordY"].values,
            "z": original["coordZ"].values - trial_df["coordZ"].values,
        }

        results_df = results_df.append(row, ignore_index=True)

    return results_df


# %%
sup_df = compute_stability_metrics(sup_df)
foundation_df = compute_stability_metrics(foundation_df)

# %% [markdown]
# ### Display input perturbations over x and y-axis

# %%
x_dist, y_dist, z_dist = [], [], []
for x_list, y_list, z_list in zip(foundation_df["x"], foundation_df["y"], foundation_df["z"]):
    x_dist += list(x_list)
    y_dist += list(y_list)
    z_dist += list(z_list)

# %%
df = {"x": x_dist, "y": y_dist, "z": z_dist}
fig = px.density_heatmap(
    df,
    "x",
    "y",
    marginal_x="histogram",
    marginal_y="histogram",
    labels={"x": "x-axis perturbation", "y": "y-axis perturbation"},
    color_continuous_scale=px.colors.sequential.Greens,
)
for trace in fig.data:
    if type(trace) == go.Histogram:
        trace.marker = dict(opacity=0.5, cauto=True, color="green")
        trace.opacity = 1

fig.update_layout(template="simple_white", width=600, height=600, xaxis=dict(tickmode="linear", tick0=0, dtick=5))

fig.write_image("figures_vector_pdf/stability_distribution.pdf")

# fig.show()

# %% [markdown]
# ### Plot prognostic stability of the feature extractor foundation model against the fine-tuned supervised model when the input seed point is perturbed, estimated through AUC for 2-year survival.

# %%
sup_df["model"] = "Supervised"
foundation_df["model"] = "Foundation model"

df = pd.concat([foundation_df, sup_df])

# %%
for metric in ["CI", "AUC"]:
    colors = ["rgb(115,115,115)", "rgb(49,130,189)"][::-1]
    gray_palette = sns.color_palette("gray", 6).as_hex()
    fig = px.box(
        df,
        x="model",
        y=metric,
        color="model",
        template="simple_white",
        width=500,
        height=600,
        points="all",
        color_discrete_sequence=[colors[0], gray_palette[1]],
    )

    min_val = min(np.min(foundation_df[metric]), np.min(sup_df[metric])) - 0.01
    fig.add_annotation(
        x=0,
        y=min_val,
        text=f"μ={np.mean(foundation_df[metric]):.2f}, σ={np.std(foundation_df[metric]):.3f}",
        showarrow=False,
        font=dict(size=10),
    )
    fig.add_annotation(
        x=1,
        y=min_val,
        text=f"μ={np.mean(sup_df[metric]):.2f}, σ={np.std(sup_df[metric]):.3f}",
        showarrow=False,
        font=dict(size=10),
    )

    fig.write_image(f"figures_vector_pdf/stability_{metric}.pdf")
    # fig.show()

# %% [markdown]
# ## Test-Retest stability


# %%
def process_rider_preds(df):
    df["ID"] = df["PatientID"].str.split("-").str[0]
    df["tag"] = df["PatientID"].str.split("-").str[1]

    pred = np.vstack([df["pred_0"].values, df["pred_1"].values])

    from scipy.special import softmax

    pred = softmax(pred, axis=0)
    df["pred_0"] = pred[0]
    df["pred_1"] = pred[1]

    df.sort_values(by="ID", inplace=True)

    return df


# %%
sup_df = pd.read_csv("../outputs/stability/supervised/rider_preds.csv")
foundation_df = pd.read_csv("../outputs/stability/foundation_features/rider_preds.csv")

# %%
foundation_df = process_rider_preds(foundation_df)
sup_df = process_rider_preds(sup_df)

# %% [markdown]
# ### Computing ICC

# %%
pg.intraclass_corr(data=foundation_df, targets="ID", raters="tag", ratings="pred_1")

# %%
pg.intraclass_corr(data=sup_df, targets="ID", raters="tag", ratings="pred_1")

# %%
results_df = pd.DataFrame()
for df, model in zip([sup_df, foundation_df], ["Supervised (Finetuned)", "Foundation (Features)"]):
    res = pg.intraclass_corr(data=df, targets="ID", raters="tag", ratings="pred_1")
    row = {
        "ICC": res.iloc[2, 2],
        "Model": model,
        "CI_low": res.iloc[2, 7][0],
        "CI_high": res.iloc[2, 7][1],
        "p": res.iloc[2, 6],
    }

    results_df = results_df.append(row, ignore_index=True)

# %%
import plotly.graph_objects as go

fig = go.Figure()
colors = ["rgb(115,115,115)", "rgb(49,130,189)"][::-1]

for i, model in enumerate(reversed(results_df["Model"].unique())):
    model_df = results_df[results_df["Model"] == model]
    fig.add_trace(
        go.Bar(
            x=model_df["Model"],
            y=model_df["ICC"],
            name=model,
            marker_color=colors[i],
            error_y=dict(
                type="data",
                symmetric=False,
                array=model_df["CI_high"] - model_df["ICC"],
                arrayminus=model_df["ICC"] - model_df["CI_low"],
            ),
        )
    )

fig.update_layout(
    title="ICC with Confidence Intervals for each Model",
    yaxis_title="Intraclass Correlation Coefficient (ICC)",
    template="simple_white",
    width=250,
    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="right", x=1),
    bargap=0,
    xaxis=dict(showticklabels=False),
    yaxis=dict(range=[0.8, 1]),  # Added this line to reduce the range of y axis
)

fig.write_image("figures_vector_pdf/icc.pdf")
# fig.show()

# %% [markdown]
# ### Compute Scatter plot of test vs retest


# %%
def process_rider_preds(df, softmax=False):
    df["ID"] = df["PatientID"].str.split("-").str[0]
    df["tag"] = df["PatientID"].str.split("-").str[1]

    if softmax:
        pred = np.vstack([df["pred_0"].values, df["pred_1"].values])

        from scipy.special import softmax

        pred = softmax(pred, axis=0)
        df["pred_0"] = pred[0]
        df["pred_1"] = pred[1]

        test_df = df[df["tag"] == "test"].sort_values(by="ID", ascending=False)["pred_1"]
        retest_df = df[df["tag"] == "retest"].sort_values(by="ID", ascending=False)["pred_1"]

    else:
        test_df = df[df["tag"] == "test"].sort_values(by="ID", ascending=False).filter(like="pred")
        retest_df = df[df["tag"] == "retest"].sort_values(by="ID", ascending=False).filter(like="pred")

    return test_df, retest_df


# %%
sup_df = pd.read_csv("../outputs/stability/supervised/rider_feats.csv")
test, retest = process_rider_preds(sup_df)
r2 = r2_score(test.values.ravel(), retest.values.ravel())
sim = stats.spearmanr(test.values.ravel(), retest.values.ravel())

colors = ["rgb(115,115,115)", "rgb(189,189,189)", "rgb(49,130,189)", "rgb(0, 163, 213)"]
gray_palette = sns.color_palette("gray", 6).as_hex()
df = pd.DataFrame({"Test": test.values.ravel(), "Retest": retest.values.ravel()})

fig = px.scatter(
    x="Test",
    y="Retest",
    data_frame=df,
    opacity=0.6,
    color_discrete_sequence=[gray_palette[1]],
    trendline="ols",
    trendline_color_override="black",
    width=600,
    height=600,
    template="simple_white",
)
# fig.show()
fig.write_image("figures_vector_pdf/rider_feats_sup.pdf")


foundation_df = pd.read_csv("../outputs/stability/foundation_features/rider_feats.csv")
test, retest = process_rider_preds(foundation_df)
r2 = r2_score(test.values.ravel(), retest.values.ravel())
sim = stats.spearmanr(test.values.ravel(), retest.values.ravel())
fig = px.scatter(
    x="Test",
    y="Retest",
    data_frame=df,
    opacity=0.6,
    color_discrete_sequence=[colors[2]],
    trendline="ols",
    trendline_color_override="black",
    width=600,
    height=600,
    template="simple_white",
)
# fig.show()
fig.write_image("figures_vector_pdf/rider_feats_ssl.pdf")

# %%
sup_df = pd.read_csv("../outputs/stability/supervised/lung1_feats.csv")
foundation_df = pd.read_csv("../outputs/stability/foundation_features/lung1_feats.csv")


# %%
def compute_feature_stability_metrics(df):
    trials = df["trial"].nunique()
    mse_items = []

    ref = df[df["trial"] == 0].filter(like="pred").values
    for n_trial in range(1, trials):
        trial_df = df[df["trial"] == n_trial]
        feats = trial_df.filter(like="pred").values

        mse = np.mean((ref - feats) ** 2)
        row = {"trial": n_trial, "MSE": mse}

        mse_items.append(row)

    return mse_items


# %%
sup_df = pd.DataFrame(compute_feature_stability_metrics(sup_df))
sup_df["model"] = "Supervised (Finetuned)"
foundation_df = pd.DataFrame(compute_feature_stability_metrics(foundation_df))
foundation_df["model"] = "Foundation (Features)"

# %%
df = pd.concat([foundation_df, sup_df])

# %%
for metric in ["MSE"]:
    colors = ["rgb(115,115,115)", "rgb(189,189,189)", "rgb(49,130,189)", "rgb(0, 163, 213)"]
    gray_palette = sns.color_palette("gray", 6).as_hex()
    fig = px.box(
        df,
        x="model",
        y=metric,
        color="model",
        template="simple_white",
        width=500,
        height=600,
        points="all",
        color_discrete_sequence=[colors[2], gray_palette[1]],
    )

    min_val = min(np.min(foundation_df[metric]), np.min(sup_df[metric])) - 0.01
    fig.add_annotation(
        x=0,
        y=min_val,
        text=f"μ={np.mean(foundation_df[metric]):.2f}, σ={np.std(foundation_df[metric]):.3f}",
        showarrow=False,
        font=dict(size=10),
    )
    fig.add_annotation(
        x=1,
        y=min_val,
        text=f"μ={np.mean(sup_df[metric]):.2f}, σ={np.std(sup_df[metric]):.3f}",
        showarrow=False,
        font=dict(size=10),
    )

    # fig.show()
    fig.write_image("figures_vector_pdf/feature_stability.pdf")


# %%
