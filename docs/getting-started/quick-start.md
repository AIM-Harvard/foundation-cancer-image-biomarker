# Quick Start


## Extracting Features from the Foundation Model (Recommended)

<u>Step 1:</u> Install all our dependencies:

```bash
pip install foundation-cancer-image-biomarker
```


Fore more info: [See detailed Install instructions](./installation.md)

<u>Step 2:</u>  Generate a CSV file with the path to your images and seed points (in physical coordinates),

| **image_path**               | **coordX** | **coordY** | **coordZ** |
|------------------------------|------------|------------|------------|
| /content/data/dummy_0.nii.gz | 55.0       | 119.0      | 27.0       |

<u>Step 3:</u>  Run this in your code environment,
```python
from fmcib.run import get_features

feature_df = get_features("csv_path_here")
```

This will preprocess your data, download our pre-trained model and execute inference on whatever device you have (CPU/GPU) and return a dataframe with the features.

!!! note
    By default, the weights are downloaded in the current working directory and are named `model_weights.torch`. You can use these downloaded weights for all future uses. If you would like to run this script in a different location and do not want to download the weights, simply copy or symlink the weights!

    If you want to download the weights separately, you can run download it from [here](https://zenodo.org/records/10528450/files/model_weights.torch?download=1) and place it in the current working directory.

You can test to check if the global coordinates are where you expect by using our provided `visualize_seed_point` function. 
We expect the coordinates in the LPS coordinate system (like ITK) but if you have it in RAS, you can negate the X and Y coordinates and that should work with our system. See [here](https://discourse.slicer.org/t/converting-fiducial-coordinates-from-ras-to-lps/9707)

```python
from fmcib.visualization.verify_io import visualize_seed_point
import pandas as pd

feature_df = pd.read_csv("csv_path_here")
visualize_seed_point(feature_df.iloc[0]) # Visualize annotations on the 0th row. 
```

## Fine-tuning the Foundation Model

There are several different ways you can fine-tune the provided foundation model.

### Using your own fine-tuning pipeline
 If you have your own fine-tuning pipeline and would like to use the model along with its weights, you can access the model with loaded weights as below,

```python
from fmcib.models import fmcib_model 

model = fmcib_model()

```

Note that for best performance, using similar data-preprocessing methods are recommended, these can be accessed using, 

```python
from fmcib.preprocessing import preprocess
from fmcib.models import fmcib_model
import pandas as pd

model = fmcib_model(eval_mode=False) # By default the model is in eval mode. Set to false if you want to train it 

df = pd.read_csv("csv_path_here")

image = preprocess(df.iloc[0])
out = model(image)

```
The above shows a full pipeline of using the transforms to get the image in the expected pre-processed input to the model. The CSV format is the same as expected for the feature extraction as shown above. 


### Using [`project-lighter`]() to fine-tune the FM
This is the easiest way to fine-tune the FM on your data. First, begin by preparing the CSV in the format similar to the feature extraction. You might have noticed by now that this format is followed across the study. Once you have this CSV, you can follow the instructions for adapting the FM in the replication guide as shown [here](../user-guide/fm_adaptation.md). Instead of using the csv path from the specific datasets in the study, provide your own csv path. 
