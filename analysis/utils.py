import numpy as np
import scipy
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import multivariate_logrank_test
from matplotlib import pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize

rcParams["font.size"] = 16
plt.rcParams["figure.dpi"] = 400


def mean_average_precision(y_true, y_pred):
    y_true = label_binarize(y_true, classes=np.unique(y_true))
    avg_precisions = [average_precision_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    return np.mean(avg_precisions)


def balanced_accuracy(y_true, y_pred):
    return balanced_accuracy_score(y_true, np.argmax(y_pred, axis=1))


def get_model_comparison_stats(y_true, pred_proba_1, pred_proba_2, nsamples=1000, fn="roc_auc_score"):
    auc_differences = []
    bootstrap_auc_differences = []

    metric1 = eval(fn)(y_true, pred_proba_1)
    metric2 = eval(fn)(y_true, pred_proba_2)
    observed_difference = metric1 - metric2

    for _ in range(nsamples):
        mask_size = pred_proba_1.shape[0] if len(pred_proba_1.shape) < 2 else (pred_proba_1.shape[0], 1)
        mask = np.random.randint(2, size=mask_size)
        p1 = np.where(mask, pred_proba_1, pred_proba_2)
        p2 = np.where(mask, pred_proba_2, pred_proba_1)
        metric1 = eval(fn)(y_true, p1)
        metric2 = eval(fn)(y_true, p2)
        auc_differences.append(metric1 - metric2)

        idx = np.random.randint(y_true.shape[0], size=y_true.shape[0])
        y_true_sample = y_true[idx]
        y_pred_sample_1 = pred_proba_1[idx]
        y_pred_sample_2 = pred_proba_2[idx]

        metric1 = eval(fn)(y_true_sample, y_pred_sample_1)
        metric2 = eval(fn)(y_true_sample, y_pred_sample_2)
        bootstrap_auc_differences.append(metric1 - metric2)

    pvalue = np.mean(np.array(auc_differences) >= observed_difference)
    diff_ci = np.percentile(bootstrap_auc_differences, (2.5, 97.5))
    return diff_ci, pvalue


def get_model_stats(y_true, y_pred, nsamples=1000, fn="roc_auc_score"):
    auc_values = []
    for _ in range(nsamples):
        idx = np.random.randint(y_true.shape[0], size=y_true.shape[0])
        y_true_sample = y_true[idx]
        y_pred_sample = y_pred[idx]
        roc_auc = eval(fn)(y_true_sample, y_pred_sample)
        auc_values.append(roc_auc)

    return np.mean(auc_values), np.percentile(auc_values, (2.5, 97.5))


def plot_km_curve(df, save_path=None, title=None):
    df = df.copy()
    fig = plt.figure(figsize=(10, 8))
    time_filter = 5
    df.loc[df["Survival.time"] / 365.0 >= time_filter, "deadstatus.event"] = 0

    ax = plt.subplot(111)
    T = df["Survival.time"] / 365.0
    E = df["deadstatus.event"]

    # Bluegreen and red colors like ggsurvplot
    colors = ["#8F1D2F", "#1D6C8E"]
    timeline = np.linspace(0, 5, 100)

    conds = []
    kmfs = []
    for idx, group in enumerate(sorted(df["group"].unique())):
        cond = df["group"] == group
        conds.append(cond)
        kmf = KaplanMeierFitter()
        kmf.fit(T[cond], event_observed=E[cond], label=group, timeline=timeline)
        kmf.plot(ax=ax, show_censors=True, ci_alpha=0.1, color=colors[idx])
        kmfs.append(kmf)

    G = df["group"]

    results = multivariate_logrank_test(T, G, event_observed=E)
    ax.set_ylim([0.1, 1.1])
    # add_at_risk_counts(*kmfs, ax=ax)
    ax.set_ylabel("Survival probability")
    ax.set_xlabel("time (years)")
    ax.legend(loc="lower left")
    if results.p_value < 0.001:
        ax.text(2, 0.9, f"n={len(df)} \nlog-rank test, p<0.001", fontstyle="italic")
    else:
        ax.text(2, 0.9, f"n={len(df)} \nlog-rank test, p={np.round(results.p_value, 3)}", fontstyle="italic")

    if title is not None:
        ax.set_title(title)

    plt.show()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)


def get_univariate_result(df):
    from lifelines import CoxPHFitter

    cph = CoxPHFitter()
    cph.fit(df, duration_col="Survival.time", event_col="deadstatus.event", formula="group")
    cph.print_summary()
    summary_dict = {
        "beta": cph.summary["coef"].to_dict()["group"],
        "HR": cph.summary["exp(coef)"].to_dict()["group"],
        "HR low CI": cph.summary["exp(coef) lower 95%"].to_dict()["group"],
        "HR high CI": cph.summary["exp(coef) upper 95%"].to_dict()["group"],
        "p.value": cph.summary["p"].to_dict()["group"],
    }

    return summary_dict
