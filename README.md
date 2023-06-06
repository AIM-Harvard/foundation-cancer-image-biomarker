# Foundation Models for Quantitative Imaging Biomarker Discovery in Cancer Imaging

<!-- <div align="center">

[![Build status](https://github.com/foundation_image_biomarker/foundation-image-biomarker/workflows/build/badge.svg?branch=master&event=push)](https://github.com/foundation_image_biomarker/foundation-image-biomarker/actions?query=workflow%3Abuild)
[![Python Version](https://img.shields.io/pypi/pyversions/foundation-image-biomarker.svg)](https://pypi.org/project/foundation-image-biomarker/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/foundation_image_biomarker/foundation-image-biomarker/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/foundation_image_biomarker/foundation-image-biomarker/blob/master/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/foundation_image_biomarker/foundation-image-biomarker/releases)
[![License](https://img.shields.io/github/license/foundation_image_biomarker/foundation-image-biomarker)](https://github.com/foundation_image_biomarker/foundation-image-biomarker/blob/master/LICENSE)
![Coverage Report](assets/images/coverage.svg)

</div> -->

## Overview
This is the the official repository for the paper <i> Foundation Models for Quantitative Imaging Biomarker Discovery for Cancer Imaging </i> <br/>
 Suraj Pai, Dennis Bontempi, Vasco Prudente, Ibrahim Hadzic, Mateo Sokač, Tafadzwa L. Chaunzwa, Simon Bernatz, Ahmed Hosny, Raymond H Mak, Nicolai J Birkbak, Hugo JWL Aerts

![](assets/images/overview.png)
 <b>General overview of the study.</b><b> a. Foundation model pre-training.</b> A foundation model, specifically a deep convolutional encoder model, was pre-trained by contrasting volumes with and without lesions. <b> b. </b> Clinical application of the foundation model. The foundation model was used to extract biomarkers and then evaluated for three classification tasks on diverse datasets. <b>c. Foundation model implementation approaches </b>  The foundation model was adapted to specific use cases by extracting features or through fine-tuning (left). <b> d. Evaluation against supervised models with selected performance metrics. </b> We compared the performance of the foundation models against conventional supervised implementations, trained from scratch (left) and fine-tuned from a different task (right). The comparison was made through several criteria for the different use cases, including quantitative performance, stability, and biological analysis. Biological, clinical, and stability analyses are limited to use case 2 due to the availability of associated data. 


## Repository Structure
```
├── foundation_image_biomarker/ # Python package for the repository. You don't need to use this unless you want to contribute to the repo
├── data/ # Download and pre-processing scripts for the data
├── experiments/ # Configuration files to reproduce the final experiments of the paper
├── models/ # Notebooks to download and test each of the final models used in the paper
└── tests/
```
## Quick Start
The easiest way to use this repository is by installing it through `pip`. This will load all the dependencies and allow you to load model weights dynamically.

```bash
pip install foundation-image-biomarker
```

Once you do this, you can look run each of our described pipelines


## Data pre-processing
You can find detailed instructions for this [here](data/README.md)

## Using all trained models
You can find detailed instructions for this [here](models/README.md)

## Re-running all the experiments outlined in the paper
You can find detailed instructions for this [here](experiments/README.md)


## 🛡 License

[![License](https://img.shields.io/github/license/foundation_image_biomarker/foundation-image-biomarker)](https://github.com/AIM-Harvard/foundation-cancer-image-biomarker/blob/master/LICENSE)

This project is licensed under the terms of the `MIT` license. See [LICENSE](https://github.com/AIM-Harvard/foundation-cancer-image-biomarker/blob/master/LICENSE) for more details.

## 📃 Citation

```bibtex
@misc{foundation-image-biomarker,
  author = {AIM-Harvard},
  title = {Official repo for "Foundation Models for Quantitative Biomarker Discovery in Cancer Imaging"},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/AIM-Harvard/foundation-cancer-image-biomarker}}
}
```
