# Model Inference

In this section, we detail how features (from the FM and pre-trained models) and predictions (from supervised models) can be extracted. 

## Extracting features from FM / pre-trained models

In order to extract features from our models, you can use the following, (at root folder location)
```bash
lighter predict --config_file ./experiments/inference/extract_features.yaml
```

!!! note
     While the above pipeline will allow you to extract features, we provide an easier and simpler, recommended API to do this. Please refer to [Quick Start](../getting-started/quick-start.md) or [Cloud Quick Start](../getting-started/cloud-quick-start.md)


However, this method might be preferred when features need to be extracted from different models (used as baselines in our study). Follow the `Note:` in the corresponding config file to change model paths and use different baselines tested.

## Running predictions from our supervised models (Finetuned FM/ Baselines)

To run predictions from our models (both supervised and self-supervised), we provide YAML files that can be run with the lighter interface. These are found in `experiments/inference`, namely `get_predictions.yaml` for getting the predictions. 

Beofre running the predictions config, if you haven't downloaded the `models` folder contraining all our baselines, you can do so using 

```bash
pip install -U "huggingface_hub[cli]"

huggingface-cli download surajpaib/fmcib models models # Run in the root of the repo

```

This will pull all the models from hugging face. Following this you can use any of these models to get predictions on the dataset of choice. 

These can be run using (at root folder location)

```bash
lighter predict --config_file ./experiments/inference/get_predictions.yaml
```
As with the previous YAMLS, please follow the 'Note:' tags to place appropriate data paths and change relevant parameters. This YAML is to be used if you want to get target predictions from the models.

!!! note
     The predictions can be extracted for different tasks as well as different baselines by following the `Note:` comments. 
