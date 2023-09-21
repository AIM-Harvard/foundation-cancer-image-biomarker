"""
This script is used to run the prediction pipeline for the foundation-cancer-image-biomarker project.

It uses a feature extractor and a linear classifier to predict the confidence score for each record in the input CSV file.
The results are saved in a new CSV file.

The script requires the following command line arguments:
- feature_extractor_weights: Path to model weights for feature extractor
- classifier_weights: Weights for linear classifier
- spatial_size: Spatial size of the volume
- csv_path: Path to CSV file
- output_path: Path to output CSV file
"""

import argparse

from fmcib.models import get_linear_classifier
from fmcib.run import get_features


def main(args):
    """
    Main function to run the prediction pipeline.

    It first extracts features from the input CSV file using the specified feature extractor.
    Then, it uses the linear classifier to predict the confidence score for each record.
    The results are saved in a new CSV file.
    """
    feature_df = get_features(args.csv_path, weights_path=args.feature_extractor_weights, spatial_size=args.spatial_size)
    pipeline = get_linear_classifier(weights_path=args.classifier_weights)
    X = feature_df.copy().filter([f"pred_{i}" for i in range(0, 4096)])
    feature_df["conf_score"] = pipeline.predict_proba(X)[:, 1]
    # Save the dataframe
    feature_df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    """
    This block is executed when the script is run directly from the command line.

    It parses the command line arguments and calls the main function.
    """
    parser = argparse.ArgumentParser(description="CLI for prediction pipeline")
    parser.add_argument(
        "--feature_extractor_weights", type=str, required=True, help="Path to model weights for feature extractor"
    )
    parser.add_argument("--classifier_weights", type=str, required=True, help="Weights for linear classifier")
    parser.add_argument("--spatial_size", type=int, nargs="+", required=True, help="Spatial size of the volume")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output CSV file")

    args = parser.parse_args()
    main(args)
