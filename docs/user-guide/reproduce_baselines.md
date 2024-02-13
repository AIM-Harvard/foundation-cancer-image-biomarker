# Reproduce Baselines

### Reproducing our baselines

We have several different baselines that we compare against in this study. 


As mentioned in [section](#supervised-models), we have three different supervised training implementations. Similar to the foundation pre-training, we use YAML files to maintain the configurations of these implementations. 

<br/>
<b> Supervised model trained from random initialization </b>

In order to reproduce this training, you can inspect the YAML configuration at `experiments/baselines/supervised_training/supervised_random_init.yaml`. By default, we configure this for Task 1. You can adapt this for Task 2 and Task 3 by searching for 'Note: ' comments in the YAML that outline what must be changed.

You can start training by running this in the root code folder,
```bash
lighter fit --config_file ./experiments/baselines/supervised_training/supervised_random_init.yaml
```

<br/>

<b> Fine-tuning a trained supervised model </b>

The YAML configuration at `experiments/baselines/supervised_training/supervised_finetune.yaml` describes how you can fine-tune an already trained supervised model. Note that this is possible only for Task 2 and Task 3 as we used the supervised model trained in Task 1 to load weights from. Make sure you download the weights for Task 1 supervised models. You can follow instructions [here](#model) 


You can start training by running this in the root code folder,
```bash
lighter fit --config_file ./experiments/baselines/supervised_training/supervised_finetune.yaml
```
<br/>


### Reproducing our linear evaluation (Logistic Regression)
