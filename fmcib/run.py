import numpy as np
import pandas as pd
import torch
from loguru import logger
from monai.networks.nets import resnet50
from tqdm import tqdm

from .models import LoadModel, fmcib_model
from .preprocessing import get_dataloader


def get_features(csv_path, weights_path=None, spatial_size=(50, 50, 50), precropped=False, **kwargs):
    """
    Extracts features from images specified in a CSV file.

    Args:
        csv_path (str): Path to the CSV file containing image paths.
        weights_path (str, optional): Path to the pre-trained weights file. Default is None.
        spatial_size (tuple, optional): Spatial size of the input images. Default is (50, 50, 50).
        precropped (bool, optional): Whether the images are already pre-cropped. Default is False.
        **kwargs: Additional arguments to be passed to the dataloader.
    Returns:
        pandas.DataFrame: DataFrame containing the original data from the CSV file along with the extracted features.
    """
    logger.info("Loading CSV file ...")
    df = pd.read_csv(csv_path)
    dataloader = get_dataloader(csv_path, spatial_size=spatial_size, precropped=precropped, **kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if weights_path is None:
        model = fmcib_model().to(device)
    else:
        logger.warning(
            "Loading custom model provided from weights file. If this is not intended, please do not provide the weights_path argument."
        )
        trunk = resnet50(
            pretrained=False,
            n_input_channels=1,
            widen_factor=2,
            conv1_t_stride=2,
            feed_forward=False,
            bias_downsample=True,
        )
        model = LoadModel(trunk=trunk, weights_path=weights_path).to(device)

    feature_list = []
    logger.info("Running inference over batches ...")

    model.eval()
    for batch in tqdm(dataloader, total=len(dataloader)):
        feature = model(batch.to(device)).detach().cpu().numpy()
        feature_list.append(feature)

    features = np.concatenate(feature_list, axis=0)
    # Flatten features into a list
    features = features.reshape(-1, 4096)

    # Add the features to the dataframe
    df = pd.concat([df, pd.DataFrame(features, columns=[f"pred_{idx}" for idx in range(4096)])], axis=1)
    return df
