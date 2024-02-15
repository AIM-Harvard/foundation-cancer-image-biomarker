import monai
import torchvision
from loguru import logger
from monai import transforms as monai_transforms

from .seed_based_crop import SeedBasedPatchCropd


def preprocess(image, spatial_size=(50, 50, 50)):
    T = get_transforms(spatial_size=spatial_size)
    return T(image)


def get_transforms(spatial_size=(50, 50, 50), precropped=False):
    if precropped:
        return monai_transforms.Compose(
            [
                monai_transforms.LoadImaged(keys=["image_path"], image_only=True),
                monai_transforms.EnsureChannelFirstd(keys=["image_path"]),
                monai_transforms.NormalizeIntensityd(keys=["image_path"], subtrahend=-1024, divisor=3072),
                monai_transforms.SelectItemsd(keys=["image_path"]),
                monai_transforms.SpatialPadd(keys=["image_path"], spatial_size=spatial_size),
                torchvision.transforms.Lambda(lambda x: x["image_path"].as_tensor()),
            ]
        )
    else:
        return monai_transforms.Compose(
            [
                monai_transforms.LoadImaged(keys=["image_path"], image_only=True, reader="ITKReader"),
                monai_transforms.EnsureChannelFirstd(keys=["image_path"]),
                monai_transforms.NormalizeIntensityd(keys=["image_path"], subtrahend=-1024, divisor=3072),
                monai_transforms.Spacingd(
                    keys=["image_path"], pixdim=1, padding_mode="zeros", mode="linear", align_corners=True, diagonal=True
                ),
                monai_transforms.Orientationd(keys=["image_path"], axcodes="LPS"),
                SeedBasedPatchCropd(
                    keys=["image_path"], roi_size=spatial_size[::-1], coord_orientation="LPS", global_coordinates=True
                ),
                monai_transforms.SelectItemsd(keys=["image_path"]),
                monai_transforms.Transposed(keys=["image_path"], indices=(0, 3, 2, 1)),
                monai_transforms.SpatialPadd(keys=["image_path"], spatial_size=spatial_size),
                torchvision.transforms.Lambda(lambda x: x["image_path"].as_tensor()),
            ]
        )


def get_dataloader(csv_path, batch_size=4, num_workers=4, spatial_size=(50, 50, 50), precropped=False):
    logger.info("Building dataloader instance ...")
    T = get_transforms(spatial_size=spatial_size, precropped=precropped)
    dataset = monai.data.CSVDataset(csv_path, transform=T)
    dataloader = monai.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return dataloader
