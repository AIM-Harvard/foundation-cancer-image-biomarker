# Pre-requisites

## Hardware
The `foundation-cancer-image-biomarker` package can operate on a standard computer hardware. For using our foundation models for inference, the system requirements are minimal as the batch size can be reduced to accomodate low memory sizes. We expect the user to have atleast 4GB of RAM and 4-cores to ensure smooth operation. A GPU with atleast 4GB of RAM will also be a plus as the inference can be ofloaded onto the GPU. 
For reproducing the training, we recommend a system with atleast 12 GB GPU VRAM, 4+ cores and 12 GB RAM. Note that batch-size, mixed-precision and number of workers can be adjusted to fit several training constraints. 


## Software
### Operating System
We have tested this system on Ubuntu 20.04 and 22.04. We expect it to work on Widows and Mac systems in inference mode without any hinderance. For reproducing the training, we support only Linux systems due to CUDA and Pytorch setups. 