## Installation

We recommend using **Python 3.8** on a Linux machine since that is the environment we have tested on. However, the use of Poetry for managing dependencies should make it compatible with Python versions above 3.8 on all platforms. We suggest using Conda to create an isolated virtual environment for the project dependencies. To download Conda, follow the instructions here: https://conda.io/projects/conda/en/latest/user-guide/install/index.html

After downloading and activating Conda, create and activate a virtual environment with the following commands:

```python
conda create -n fmcib_env python=3.8
conda activate fmcib_env
```

The most straightforward way to use this repository is by installing our publicly available `pip` package, which includes the majority of dependencies needed for reproducing the experiments and lets you load model weights dynamically.

Download our latest pre-release version with the following command:

```bash
pip install foundation-cancer-image-biomarker --pre
```

If you'd like to download and process the data from scratch, install the additional required dependencies using:

```bash
pip install -r additional_requirements.txt
```

You can verify if the dependencies are installed by: 
```bash
python -c "import fmcib; print(fmcib.__version__)"
```
