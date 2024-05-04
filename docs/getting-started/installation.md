# Installation

Our package is offered through a very simple pip install, you can run:
```bash
pip install foundation-cancer-image-biomarker
```
The package provides simple ways to interact with the foundation model through the means of several utility scripts and functions. 

If you would like to install the bleeding edge version, please add 
```bash
pip install foundation-cancer-image-biomarker --pre
```

Once you have installed the package, you can move to our [Quick Start](./quick-start.md) guide.


## Notes
We recommend using **Python 3.8** on a Linux machine since that is the environment we have tested on. However, we use Poetry to manage our dependencies which should make it compatible with Python versions above 3.8 on all platforms. We suggest using Conda to create an isolated virtual environment for the project dependencies. To download Conda, follow the instructions here: https://conda.io/projects/conda/en/latest/user-guide/install/index.html

### Hardware
The `foundation-cancer-image-biomarker` package can operate on a standard computer hardware. For using our foundation models for inference, which should be most of the use-cases, the system requirements are minimal as the batch size can be reduced to accomodate low memory sizes. We expect the user to have atleast 4GB of RAM and 4-cores to ensure smooth operation. A GPU with atleast 4GB of RAM will also be a plus as the inference can be ofloaded onto the GPU. 

For reproducing the training, we recommend a system with atleast 12 GB GPU VRAM, 4+ cores and 12 GB RAM. Note that batch-size, mixed-precision and number of workers can be adjusted to fit several training constraints. 


### Software
We have tested this system on Ubuntu 20.04 and 22.04. We expect it to work on Widows and Mac systems in inference mode without any hinderance. For reproducing the training, we support only Linux systems due to CUDA and Pytorch setups. 
