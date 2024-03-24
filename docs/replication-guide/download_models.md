# Download Trained Models

All our models are will be made available to the public through Zenodo upon publication. Currently, we release these using Dropbox for the reviewers to use and test. Scripts for downloading these models are present under `models`. 

As part of our study we develop and share the following,

### Self-supervised pre-training model
We developed the pretrained model using the DeepLesion dataset with 11,467 annotated CT lesions identified from 2,312 unique patients. Lesion findings were diverse and included multiple lesions, such as lung nodules, cysts, and breast lesions, among numerous others. A task-agnostic contrastive learning strategy was used to pre-train the model on these lesion findings. Refer to the methods section for more information or the reproducing our models section.

To download these models run,
````bash
cd models
bash download_foundation_pretrained_model.sh
````

You can also extract the dropbox links and place them in the target location mentioned.



<div style="display: flex; justify-content: center"><img src="../../assets/images/ssl_pretraining.png" width=600 /></div>


The pre-trained model is implemented on downstreams task using supervised training or linear evaluation approaches. For these we develop,

### Supervised models

We developed several baseline training approaches,

<details><summary> Supervised model trained from random initialization </summary>

<br>
<div style="display: flex; justify-content: center"><img src="../../assets/images/implementation1.png" width=400 /></div>

</details>


<details><summary> Fine-tuning a trained supervised model </summary>

<br>
<div style="display: flex; justify-content: center"><img src="../../assets/images/implementation2.png" width=400 /></div>

</details>

<details><summary> Fine-tuning a pre-trained foundation model </summary>

<br>
<div style="display: flex; justify-content: center"><img src="../../assets/images/implementation3.png" width=400 /></div>

</details>



To download these models run,
````bash
cd models
bash download_task1_baselines.sh
bash download_task2_baselines.sh
bash download_task3_baselines.sh


````


### Linear (Logistic Regression) models
Our linear model takes features extracted from the pre-trained foundation model and builds a logistic regression classifer to predict outcome. 
<div style="display: flex; justify-content: center"><img src="../../assets/images/linear_model.png" width=400 /></div>

&emsp;
To download these models run,
````bash
cd models
bash download_linear_models.sh
````

These models can also be found at this [link](https://www.dropbox.com/scl/fo/brhqokhzn839zez15erzf/h?dl=0&rlkey=wzvgrobl8p3v49ettm16uxbyy). In addition to providing our models, we also provide comprehensive documentation and ongoing support to users through [project-lighter](https://zenodo.org/record/8007711) to reproduce our results and workflows.
