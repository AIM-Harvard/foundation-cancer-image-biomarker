import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import multivariate_logrank_test
from matplotlib import pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import auc, average_precision_score, balanced_accuracy_score, roc_curve

rcParams["font.size"] = 16
plt.rcParams["figure.dpi"] = 400


def average_precision(target, pred, n_classes):
    """ "
    Function to calculate average precision for a given target and prediction. Handles cases when prediction is multiclass
    and target is provided as class labels
    """
    if n_classes == 1:
        return average_precision_score(target, pred)
    else:
        per_class_scores = []
        for class_idx in range(n_classes):
            binarized_target = [1 if t == class_idx else 0 for t in target]
            score = average_precision_score(binarized_target, pred[class_idx])
            per_class_scores.append(score)
        return per_class_scores


def auc_roc(target, pred, n_classes):
    """ "
    Function to calculate AUC precision for a given target and prediction. Handles cases when prediction is multiclass
    and target is provided as class labels
    """
    if n_classes == 1:
        fpr, tpr, _ = roc_curve(target, pred)
        return auc(fpr, tpr)
    else:
        per_class_scores = []
        for class_idx in range(n_classes):
            (
                fpr,
                tpr,
                _,
            ) = roc_curve(target, pred[class_idx], pos_label=class_idx)
            score = auc(fpr, tpr)
            per_class_scores.append(score)
        return per_class_scores


def get_ci(time, pred, event, sample_target=True, axis=-1):
    import lifelines

    ci_values = []
    if len(pred.shape) > 1:
        for sample in range(pred.shape[0]):
            T = time[sample] if sample_target else time
            E = event[sample] if sample_target else event
            ci = lifelines.utils.concordance_index(T, pred[sample], E)
            ci_values.append(ci)
    else:
        return lifelines.utils.concordance_index(time, pred, event)

    return np.array(ci_values)


def balanced_acc(target, pred, _):
    """ "
    Function to calculate AUC precision for a given target and prediction. Handles cases when prediction is multiclass
    and target is provided as class labels
    """
    pred_label = np.argmax(pred, axis=0)
    return balanced_accuracy_score(target, pred_label)


def get_score(target, pred, fn="average_precision", sample_target=True, n_classes=1, axis=0):
    """ "
    Function to calculate a given metric/statistic for a given target and prediction.
    It is assumed that the prediction is a probability distribution over classes.
    The function is designed to work with `scipy.stats.bootstrap` function
    """
    expected_pred_dim = 1 if n_classes == 1 else 2
    if len(pred.shape) > expected_pred_dim:
        if n_classes != 1:
            class_axis = (np.array(pred.shape) == n_classes).nonzero()[0][0]
            pred = np.swapaxes(pred, 0, class_axis)
            target = np.swapaxes(target, 0, class_axis) if sample_target else target

        resamples_axes = 1 if n_classes > 1 else 0
        n_resamples = pred.shape[resamples_axes]
        bootstrap_scores = []
        for sample in range(n_resamples):
            score = eval(fn)(
                target[sample] if sample_target else target, np.take(pred, sample, axis=resamples_axes), n_classes
            )
            mean_score = np.mean(score)
            bootstrap_scores.append(mean_score)

        return np.array(bootstrap_scores)

    # General case
    else:
        if axis != -1:
            pred = np.swapaxes(pred, -1, axis)
        metric_scores = eval(fn)(target, pred, n_classes)
        return np.mean(metric_scores)


def get_score_difference(target, pred1, pred2, fn="average_precision", sample_target=True, n_classes=1, axis=0):
    """ "
    Function to calculate a difference metric/statistic for a given target and two predictions.
    In our case, we compare predictions from two different model implementation
    It is assumed that the predictions are a probability distribution over classes.
    The function is designed to work with `scipy.stats.bootstrap` and `scipy.stats.permutation_test` functions
    """
    score1 = get_score(target, pred1, fn, sample_target, n_classes, axis)
    score2 = get_score(target, pred2, fn, sample_target, n_classes, axis)
    return score1 - score2


def get_ci_differences(time, event, pred1, pred2, sample_target=True, axis=0):
    """ "
    Function to calculate CI differences for two predictions.
    In our case, we compare predictions from two different model implementation
    It is assumed that the predictions are a probability distribution over classes.
    The function is designed to work with `scipy.stats.bootstrap` and `scipy.stats.permutation_test` functions
    """
    ci1 = get_ci(time, pred1, event, sample_target, axis)
    ci2 = get_ci(time, pred2, event, sample_target, axis)
    return ci1 - ci2


def plot_km_curve(df):
    fig = plt.figure(figsize=(10, 8))
    time_filter = 5
    df.loc[df["Survival.time"] / 365.0 >= time_filter, "deadstatus.event"] = 0

    ax = plt.subplot(111)
    T = df["Survival.time"] / 365.0
    E = df["deadstatus.event"]

    timeline = np.linspace(0, 5, 100)

    conds = []
    kmfs = []
    for group in sorted(df["group"].unique()):
        cond = df["group"] == group
        conds.append(cond)
        kmf = KaplanMeierFitter()
        kmf.fit(T[cond], event_observed=E[cond], label=group, timeline=timeline)
        kmf.plot(ax=ax, show_censors=True, ci_alpha=0.1)
        kmfs.append(kmf)

    G = df["group"]

    results = multivariate_logrank_test(T, G, event_observed=E)
    ax.set_ylim([0.1, 1.1])
    # add_at_risk_counts(*kmfs, ax=ax)
    ax.set_ylabel("Survival probability")
    ax.set_xlabel("time (years)")
    ax.legend(loc="lower left")
    ax.text(2, 0.9, f"n={len(df)} \nlog-rank test, p={np.round(results.p_value, 3)}", fontstyle="italic")
    plt.show()


def get_univariate_result(df):
    from lifelines import CoxPHFitter

    cph = CoxPHFitter()
    cph.fit(df, duration_col="Survival.time", event_col="deadstatus.event", formula="group")
    cph.print_summary()
