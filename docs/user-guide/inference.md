## Model Inference


### Running predictions from our models.

To run predictions from our models (both supervised and self-supervised), we provide YAML files that can be run with the lighter interface. These are found in `experiments`, namely `get_predictions.yaml` for getting the predictions. 

These can be run using (at root folder location)

```bash
lighter predict --config_file ./experiments/get_predictions.yaml
```
As with the previous YAMLS, please follow the 'Note:' tags to place appropriate data paths and change relevant parameters. This YAML is to be used if you want to get target predictions from the models.

### Extracting features from our models.

In order to extract features from our models, you can use the following, (at root folder location)
```bash
lighter predict --config_file ./experiments/extract_features.yaml
```


!!! note
     While the above pipeline will allow you to extract features, we provide an easier and simpler way to do this through Google Colab. To promote accessibility to our work, we have simplified the feature extraction process and provide the notebook [here](https://colab.research.google.com/drive/1JMtj_4W0uNPzrVnM9EpN1_xpaB-5KC1H?usp=sharing)