import SimpleITK as sitk
from platipy.imaging import ImageVisualiser


def visualize_seed_point(row):
    image = sitk.ReadImage(row["image_path"])
    image_centroid = image.TransformPhysicalPointToContinuousIndex([row["coordX"], row["coordY"], row["coordZ"]])
    image_centroid = [int(x) for x in image_centroid]
    visualiser = ImageVisualiser(image, cut=image_centroid[::-1], window=(-1000, 2048))
    visualiser.add_bounding_box([*image_centroid, 1, 1, 1], linewidth=5)
    visualiser.show()
