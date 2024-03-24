# Reproducing Foundation Model Training

## Data Setup for the Models
Make sure you download all the datasets before starting to train. If you are using your own datasets, then you will need to format them according to the structure as described to ensure the easiest translation to using our pipeline. 


Our framework ingests datasets as CSV files with `image_path` column providing location of the image (on your system) to be used, `coordX`, `coordY` and `coordZ` providing the <b>global coordinates</b> of the seed point around which a patch is cropped. 

We crop a `[50, 50, 50]` patch around the seed point. Please refer to our paper for more details on this. 


For supervised fine-tuning, along with these columns, label columns are needed as below,
```
Task 1: Coarse_lesion_type
Task 2: malignancy
Task 3: survival
```


## Reproducing the FM pre-training
The crux of our study is the self-supervised/weakly supervised pre-training procedure. We implemented contrastive pre-training using a modified version of the SimCLR framework. The SimCLR framework's general principle involves transforming a single data piece (e.g., a patch taken from a CT scan) into two correlated and augmented samples (e.g., the same patch rotated 15 degrees clockwise and flipped horizontally). A convolutional encoder is then used to extract latent representations from these samples. Through a contrastive loss function, the model learns to identify similar representations from the same data sample and dissimilar representations from different data samples. The framework emphasizes effective transformation choices, convolutional encoder architectures, and contrastive loss functions for optimal self-supervised learning performance. To effectively represent the nature of medical images, we made modifications to each of these components. 

1. Medical image specific transformations implemented from [Project-MONAI](https://monai.io/) and custom implementations at `fmcib.ssl.transforms`
2. 3D ResNet from Project-MONAI
3. Custom implemented modified loss function and SimCLR architecture that can be found under `fmcib.ssl.losses.NTXentNegativeMinedLoss` and `fmcib.ssl.modules.ExNegSimCLR`

We use project-lighter developed internally within our lab to provide reproducible training for all the models used in this study. [Project-lighter](https://github.com/project-lighter/lighter) allows a YAML-based configuration system along with a python-based CLI to allow quick, easy and scalable experimentation.

To pre-train on the DeepLesion pretraining dataset, you can find the YAML for the pre-training at `experiments/pretraining/fmcib_pretrain.yaml`. It is assumed that you have a GPU available. If you do not (not recommended and not tested and probably takes forever), then edit the following parameters
```yaml
  accelerator: cpu
  strategy: auto
  devices: 1
``` 

The default training assumes 2 GPUs as mentioned in the paper. You can change this by setting to your desired number of GPUs here:
```yaml
devices: 1
```

Change the path of the train dataset to the pre-train set generated earlier in the data-preprocessing step. Read the [data](data.md) section for more information.

```yaml
datasets:
  train:
    _target_: fmcib.ssl.datasets.SSLRadiomicsDataset
    path: "your_pretrain_set_path_goes_here" 
```

Now you can start training by running this in the root code folder,


```bash
lighter fit --config_file ./experiments/pretraining/fmcib_pretrain.yaml
```