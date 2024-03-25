# Reproduce Baselines

Reproducing baselines used in this study is very similar to the adaptation of the FM as essentially we are just using the FM weights for adaptation. 

## Randomly initialized

We provide the YAML configuration to train the random init baseline at `experiments/baselines/supervised_training/supervised_random_init.yaml`

By default, we configure this for Task 1. You can adapt this for Task 2 and Task 3 by searching for `Note: ` comments in the YAML that outline what must be changed. 

You can start training by running this in the root code folder,
```bash
lighter fit --config_file ./experiments/baselines/supervised_training/supervised_random_init.yaml
```
 
## Transfer learning
We provide the YAML configuration to train the transfer learning baseline at `experiments/baselines/supervised_training/supervised_finetune.yaml`

This baseline is only used for Task 2 and Task 3 as we use the random init baseline from Task 1 for the transfer. Follow the `Note: ` comments to switch between Task 2 and Task 3 configurations. 

You can start training by running this in the root code folder,
```bash
lighter fit --config_file ./experiments/baselines/supervised_training/supervised_finetune.yaml
```

## Med3D / MedicalNet
Original repo: https://github.com/Tencent/MedicalNet

We have provided re-implementations of Med3D to fit into our YAML workflows at `experiments/baselines/med3d/finetune.yaml`. Again, the `Note: ` comments help adapt for different tasks. 


You can start training by running this in the root code folder,
```bash
lighter fit --config_file ./experiments/baselines/med3d/finetune.yaml
```

## Models Genesis
Original repo: https://github.com/MrGiovanni/ModelsGenesis

We have provided re-implementations of Models Genesis to fit into our YAML workflows at `experiments/baselines/models_genesis/finetune.yaml`. Again, the `Note: ` comments help adapt for different tasks. 


You can start training by running this in the root code folder,
```bash
lighter fit --config_file ./experiments/baselines/models_genesis/finetune.yaml
```
