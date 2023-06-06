import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from .models import resnet50
from .preprocessing import get_dataloader


def get_features(csv_path):
    logger.info("Loading CSV file ...")
    df = pd.read_csv(csv_path)
    dataloader = get_dataloader(csv_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = resnet50().to(device)

    feature_list = []
    logger.info("Running inference over batches ...")

    for batch in tqdm(dataloader, total=len(dataloader)):
        feature = model(batch.to(device)).detach().cpu().numpy()
        feature_list.append(feature)

        if len(feature_list) == 1:
            break

    features = np.concatenate(feature_list, axis=0)
    # Flatten features into a list
    features = features.reshape(-1, 4096)

    # Add the features to the dataframe
    df = pd.concat([df, pd.DataFrame(features)], axis=1)
    return df
