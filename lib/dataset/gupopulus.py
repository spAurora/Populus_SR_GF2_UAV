from .image_folders import ImageFolders


class GUPOPULUSDataset(ImageFolders):
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
                # data_dir / "flickr2k" / "Flickr2K_HR",  # wHy 241008
            ]
            # print('random crop size:', random_crop_size) # wHy 241008
        else:
            paths = [
                data_dir / "gupopulus_2" / "gupopulus_valid_HR",
                data_dir / "gupopulus_2" / "gupopulus_valid_LR_bicubic" / "X2",
            ]

        print('downscale_factor:', downscale_factor)

        super().__init__(
            [paths[0]],
            [paths[1]],
            downscale_factor,
            random_crop_size=random_crop_size,
            deterministic=deterministic,
            repeat=repeat,
        )
