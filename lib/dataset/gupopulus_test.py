import os
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from .utils import image_to_tensor

class GUPOPULUSTestDataset(Dataset):
    def __init__(self, data_dir):
        """
        Dataset for testing GUPOPULUS images.

        Args:
            data_dir (str or Path): Root directory containing the image folders.
        """
        self.hr_dir = Path(data_dir) / "gupopulus_2" / "gupopulus_valid_HR"
        self.lr_dir = Path(data_dir) / "gupopulus_2" / "gupopulus_valid_LR_bicubic" / "X2"
        
        # 获取 HR 和 LR 文件夹下所有的 PNG 文件名，确保按文件名排序
        self.hr_images = sorted([f for f in os.listdir(self.hr_dir) if f.endswith('.png')])
        # self.lr_images = sorted([f for f in os.listdir(self.lr_dir) if f.endswith('.png')])

        self.lr_images = []
        for hr_image in self.hr_images:
            self.lr_images.append(hr_image[:-4]+'x'+os.path.split(self.lr_dir)[-1][-1]+'.png')

        print(self.hr_images)
        print(self.lr_images)

        # 检查 HR 和 LR 图像数量是否一致
        assert len(self.hr_images) == len(self.lr_images), "HR and LR image counts do not match."

    def __len__(self):
        """
        Return the number of image pairs.
        """
        return len(self.hr_images)

    def __getitem__(self, idx):
        """
        Get a pair of HR and LR images.

        Args:
            idx (int): Index of the image pair.

        Returns:
            dict: A dictionary with 'image' for HR image tensor and 'image_lr' for LR image tensor.
        """
        # 获取当前索引对应的 HR 和 LR 文件名
        hr_filename = self.hr_images[idx]
        lr_filename = self.lr_images[idx]

        # 构建完整的 HR 和 LR 图像路径
        hr_path = self.hr_dir / hr_filename
        lr_path = self.lr_dir / lr_filename

        # 读取并转换 HR 和 LR 图像为 numpy 数组
        with Image.open(hr_path) as hr_image:
            image_hr = np.asarray(hr_image)
        with Image.open(lr_path) as lr_image:
            image_lr = np.asarray(lr_image)

        # 将 numpy 数组转换为张量
        image_hr = image_to_tensor(image_hr)
        image_lr = image_to_tensor(image_lr)

        return {"image": image_hr, "image_lr": image_lr, "image_name_hr":hr_filename}
