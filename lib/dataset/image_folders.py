import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .imresize import imresize
from .utils import image_to_hr_lr_tensor, image_to_tensor

from pathlib import Path

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif")


class ImageFolders(Dataset):
    def __init__(
        self,
        hr_paths,  # 高分辨率图像文件夹路径
        lr_paths,  # 低分辨率图像文件夹路径
        downscale_factor,
        *,
        random_crop_size=None,
        deterministic=False,
        repeat=1,
        pre_resize=None,
        pre_crop=False,
    ):
        
        self.hr_files = []
        self.lr_files = []

        # print(hr_paths, lr_paths)
        # # 获取高分辨率图像的文件路径
        # for hr_path in hr_paths:
        #     hr_paths = [
        #         file
        #         for file in hr_path.iterdir()
        #         if file.is_file() and str(file).lower().endswith(IMAGE_EXTENSIONS)
        #     ]
        #     hr_paths.sort()
        #     self.hr_files.extend(hr_paths)
        # print(self.hr_files)

        # # 获取低分辨率图像的文件路径
        # for lr_path in lr_paths:
        #     lr_paths = [
        #         file
        #         for file in lr_path.iterdir()
        #         if file.is_file() and str(file).lower().endswith(IMAGE_EXTENSIONS)
        #     ]
        #     lr_paths.sort()
        #     self.lr_files.extend(lr_paths)
        # print(self.lr_files)

        print(hr_paths, lr_paths)
        # 获取高分辨率图像的文件路径
        for i in range(len(hr_paths)):
            hr_path = hr_paths[i]
            for file in hr_path.iterdir():
                if file.is_file() and str(file).lower().endswith(IMAGE_EXTENSIONS):
                    self.hr_files.append(Path(str(hr_path)+'/'+file.name))
                    self.lr_files.append(Path(str(lr_paths[i])+'/'+file.name[0:-4]+'x'+str(downscale_factor)+file.name[-4:]))
        print(self.hr_files)
        print(self.lr_files)

        # 检查高分辨率和低分辨率图像的数量是否一致
        assert len(self.hr_files) == len(self.lr_files), (
            f"High resolution and low resolution images count mismatch: "
            f"{len(self.hr_files)} HR images, {len(self.lr_files)} LR images"
        )        

        self.downscale_factor = downscale_factor
        self.random_crop_size = random_crop_size
        self.deterministic = deterministic
        self.repeat = repeat
        self.pre_resize = pre_resize
        self.pre_crop = pre_crop

        if self.deterministic:
            g = torch.Generator()
            g.manual_seed(123456789)
            self.crop_indices_frac = torch.rand(
                size=(len(self.hr_files) * self.repeat, 2),
                dtype=torch.float64,
                generator=g,
            )
        else:
            self.crop_indices_frac = None

    def __len__(self):
        return len(self.hr_files) * self.repeat

    def __getitem__(self, idx):
        # 获取高分辨率图像的路径
        hr_path = self.hr_files[idx % len(self.hr_files)]
        lr_path = self.lr_files[idx % len(self.lr_files)]
        
        # 检查低分辨率图像是否存在
        if not lr_path.exists():
            raise FileNotFoundError(f"Low-resolution image not found: {lr_path}")
        
        # 读取高分辨率图像
        with Image.open(hr_path) as hr_image_file:
            hr_image = hr_image_file.convert("RGB")
            hr_image = np.asarray(hr_image)

        # 读取低分辨率图像
        with Image.open(lr_path) as lr_image_file:
            lr_image = lr_image_file.convert("RGB")
            lr_image = np.asarray(lr_image)

        # print(hr_image.shape, lr_image.shape)

        # 预裁剪（如果需要）wHy241010 有bug
        if self.pre_crop:
            target_size = min(hr_image.shape[0], hr_image.shape[1])
            if self.deterministic:
                start_x = int(round((hr_image.shape[0] - target_size) / 2.0))
                start_y = int(round((hr_image.shape[1] - target_size) / 2.0))
            else:
                start_x = np.random.randint(0, hr_image.shape[0] - target_size + 1)
                start_y = np.random.randint(0, hr_image.shape[1] - target_size + 1)
            
            # 同步裁剪高分辨率和低分辨率图像
            hr_image = hr_image[start_x : start_x + target_size, start_y : start_y + target_size, :]
            lr_image = lr_image[start_x : start_x + target_size, start_y : start_y + target_size, :]

        # 调整高分辨率图像大小（如果需要）
        if self.pre_resize is not None:
            hr_image = imresize(hr_image, output_shape=self.pre_resize)
            # 调整低分辨率图像大小，确保其与高分辨率图像保持比例
            lr_resize_shape = (
                self.pre_resize[0] // self.downscale_factor,
                self.pre_resize[1] // self.downscale_factor,
            )
            lr_image = imresize(lr_image, output_shape=lr_resize_shape)

        # 随机裁剪图像（如果指定了随机裁剪尺寸）
        if self.random_crop_size is not None:
            x_idx_limit = hr_image.shape[0] - self.random_crop_size + 1
            y_idx_limit = hr_image.shape[1] - self.random_crop_size + 1

            if not self.deterministic:
                x_idx = torch.randint(0, x_idx_limit, size=()).item()
                y_idx = torch.randint(0, y_idx_limit, size=()).item()
            else:
                x_idx = (
                    (self.crop_indices_frac[idx, 0] * x_idx_limit)
                    .floor()
                    .int()
                    .clamp(0, x_idx_limit - 1)
                    .item()
                )
                y_idx = (
                    (self.crop_indices_frac[idx, 1] * y_idx_limit)
                    .floor()
                    .int()
                    .clamp(0, y_idx_limit - 1)
                    .item()
                )

            # 同步裁剪高分辨率和低分辨率图像
            hr_image = hr_image[
                x_idx : x_idx + self.random_crop_size,
                y_idx : y_idx + self.random_crop_size,
                :,
            ]
            
            # 计算低分图像的裁剪尺寸
            lr_crop_size = self.random_crop_size // self.downscale_factor

            # 裁剪低分辨率图像，使用相同的 x_idx 和 y_idx
            lr_image = lr_image[
                x_idx // self.downscale_factor : (x_idx + self.random_crop_size) // self.downscale_factor,
                y_idx // self.downscale_factor : (y_idx + self.random_crop_size) // self.downscale_factor,
                :,
            ]

        # 将图像转换为张量格式
        hr_image = image_to_tensor(hr_image)
        lr_image = image_to_tensor(lr_image)

        # print(hr_image.shape, lr_image.shape)

        return {"image": hr_image, "image_lr": lr_image}
