import monai
import torchvision
from loguru import logger
from monai import transforms as monai_transforms

from .seed_based_crop import SeedBasedPatchCropd

monai.data.set_track_meta(True)

T = monai_transforms.Compose(
    [
        monai_transforms.LoadImaged(keys=["image_path"], image_only=True, reader="ITKReader"),
        monai_transforms.EnsureChannelFirstd(keys=["image_path"]),
        monai_transforms.Spacingd(keys=["image_path"], pixdim=1, mode="bilinear", align_corners=True, diagonal=True),
        monai_transforms.NormalizeIntensityd(keys=["image_path"], subtrahend=-1024, divisor=3072),
        monai_transforms.Orientationd(keys=["image_path"], axcodes="LPS"),
        SeedBasedPatchCropd(keys=["image_path"], roi_size=(50, 50, 50), coord_orientation="LPS", global_coordinates=True),
        monai_transforms.SelectItemsd(keys=["image_path"]),
        monai_transforms.Transposed(keys=["image_path"], indices=(0, 3, 2, 1)),
        monai_transforms.SpatialPadd(keys=["image_path"], spatial_size=(50, 50, 50)),
        torchvision.transforms.Lambda(lambda x: x["image_path"].as_tensor()),
    ]
)


def preprocess(image):
    return T(image)


def get_transforms():
    return T


def get_dataloader(csv_path, batch_size=4, num_workers=4):
    logger.info("Building dataloader instance ...")
    dataset = monai.data.CSVDataset(csv_path, transform=T)
    dataloader = monai.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return dataloader
