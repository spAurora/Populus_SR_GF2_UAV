from .image_folders import ImageFolders_Mask, ImageFolders


class GUPOPULUSDataset_MASK(ImageFolders_Mask):
    def __init__(
        self,
        data_dir,
        downscale_factor,
        train: bool,
        *,
        random_crop_size=None,
        deterministic=False,
        repeat=1
    ):
        if train:
            paths = [
                data_dir / "gupopulus_2" / "gupopulus_train_HR",
                data_dir / "gupopulus_2" / "gupopulus_train_LR",
                data_dir / "gupopulus_2" / "gupopulus_train_MASK",
            ]

            super().__init__(
                [paths[0]],
                [paths[1]],
                [paths[2]],
                downscale_factor,
                random_crop_size=random_crop_size,
                deterministic=deterministic,
                repeat=repeat,
            )
        else:
            raise ValueError("Train should be true.")


class GUPOPULUSDataset_MASK_VAL(ImageFolders):
    def __init__(
        self,
        data_dir,
        downscale_factor,
        train: bool,
        *,
        random_crop_size=None,
        deterministic=False,
        repeat=1
    ):
        if not train:    
            paths = [
                data_dir / "gupopulus_2" / "gupopulus_valid_HR",
                data_dir / "gupopulus_2" / "gupopulus_valid_LR",
            ]

            super().__init__(
                [paths[0]],
                [paths[1]],
                downscale_factor,
                random_crop_size=random_crop_size,
                deterministic=deterministic,
                repeat=repeat,
            )
        else:
            raise ValueError("Train should be false.")
