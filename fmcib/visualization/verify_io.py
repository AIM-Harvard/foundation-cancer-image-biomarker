import SimpleITK as sitk
from platipy.imaging import ImageVisualiser


def visualize_seed_point(row):
    """
    Visualizes a seed point on an image.

    Args:
        row (pandas.Series): A row containing the information of the seed point, including the image path and the coordinates.
        The following columns are expected: "image_path", "coordX", "coordY", "coordZ".

    Returns:
        None
    """
    image = sitk.ReadImage(row["image_path"])
    image_centroid = image.TransformPhysicalPointToContinuousIndex([row["coordX"], row["coordY"], row["coordZ"]])
    image_centroid = [int(x) for x in image_centroid]
    visualiser = ImageVisualiser(image, cut=image_centroid[::-1], window=(-1000, 2048))
    visualiser.add_bounding_box([*image_centroid, 1, 1, 1], linewidth=5)
    visualiser.show()
