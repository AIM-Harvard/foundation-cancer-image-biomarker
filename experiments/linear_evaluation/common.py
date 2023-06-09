from random import random
import sklearn
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score
import optuna
from functools import partial
from sklearn.utils.extmath import softmax
import numpy as np
SEED = 42

# Optuna hyperparameter tuning
def objective(trial, train_X, train_y, val_X=None, val_y=None, scoring="roc_auc"):
    # Define hyperparameters
    C = trial.suggest_loguniform("C", 1e-6, 1e3)
    # Define classifier
    classifier = sklearn.linear_model.LogisticRegression(C=C, random_state=SEED, max_iter=1000)
    
    if val_X is None:
        scores = cross_val_score(classifier, train_X, train_y, cv=10, scoring="roc_auc")
        score = scores.mean()

    else:
        classifier.fit(train_X, train_y)
        
        # Handle scoring when AUC
        if scoring == "roc_auc":
            preds = classifier.predict_proba(val_X)[:, 1]
            score = sklearn.metrics.roc_auc_score(val_y, preds)
        else:
            score = classifier.score(val_X, val_y)

    return score
    
def optuna_hyperparameter_tuning(train_X, train_y, val_X, val_y, scoring, trials=100):
    study = optuna.create_study(direction="maximize")
    optuna.logging.set_verbosity(optuna.logging.DEBUG)
    study.optimize(partial(objective, train_X=train_X, train_y=train_y, \
                            val_X=val_X, val_y=val_y, scoring=scoring), n_trials=trials)

    return study.best_params

