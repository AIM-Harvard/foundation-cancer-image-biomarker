# Adaptation of the FM

The FM was adapted by either fine-tuning all its weights or by freezing its weights and adding a linear model on top. 

## Adaptation through fine-tuning

We provide the YAML configuration for this at `experiments/adaptation/fmcib_finetune.yaml`.

By default, we configure this for Task 1. You can adapt this for Task 2 and Task 3 by searching for `Note: ` comments in the YAML that outline what must be changed. Make sure you download the weights for the pre-trained foundation model before attempting to reproduce this training. 


You can start training by running this in the root code folder,
```bash
lighter fit --config_file=./experiments/adaptation/fmcib_finetune.yaml
```

## Adaptation through linear evaluation
4096 features from the FM were for each data point and used to train a logistic regression model using the scikit-learn framework. A comprehensive parameter search for the logistic regression model was performed using the [Optuna](https://optuna.org/) hyper-parameter optimization framework. The code and utilities for performing the logistic regression modelling is provided in `experiments/adaptation/linear`

In order to perform the modelling, you can run 
```bash

cd experiments/adaptation
python run.py <features_folder> <label>
```

The <features_folder> must contain `train_features.csv`, `val_features.csv` and `test_features.csv` all extracted from our foundation model. The process to extract features from our foundation model is highlighted in this [section](#running-predictions-and-extracting-features-from-our-models)

The <label> corresponds to the column in the csv files that contains the supervised label to predict. For example, in use-case 2 the label is `malignancy`. 

You can also provide scoring metrics, for instance,  using `--scoring roc_auc` where the scoring metric is a sklearn scorer. You can also provide the number of trials the optimization framework needs to be run for using `--trials`. 

The features folder is provided under `outputs/foundation_features` to try our the modelling process. Refer [here](#feature-extaction-pipeline)