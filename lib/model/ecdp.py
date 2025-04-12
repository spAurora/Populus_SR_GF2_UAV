import torch
import torchvision.transforms.functional as FV
from torch import nn

from .diffusion import Diffusion
from .utils import ImageSizeMixin
from .rrdbnet import RRDBNet
from .unet import UNet
import numpy as np
from PIL import Image
import os
from osgeo import gdal

class ECDP(ImageSizeMixin, nn.Module):
    def __init__(self, options):
        super().__init__()

        self.in_channels = options.model.input_channels
        self.diffusion = Diffusion(
            UNet(
                self.in_channels,
                128,
                options.model.rrdb_channels,
                out_channels=2 * self.in_channels,
            )
        )

        # TODO: load pretrained rrdb
        # you should place the parameters of pretrained RRDBNet to the
        # files here
        rrdbnet = RRDBNet(options, rrdb_channels=options.model.rrdb_network_features)
        if options.train.dataset == "df2k":
            t = torch.load(
                "pretrained-rrdbnet-df2k.pt", map_location=torch.device("cpu"), weights_only=True
            )
        elif options.train.dataset == "gupopulus":
            t = torch.load(
                "pretrained-rrdbnet-gupopulus.pt", map_location=torch.device("cpu"), weights_only=True
            )
        elif options.train.dataset == "parcel_s2":
            t = torch.load(
                "pretrained-rrdbnet-parcel_s2.pt", map_location=torch.device("cpu"), weights_only=True
            )
        elif options.train.dataset == "parcel_gf2":
            t = torch.load(
                "pretrained-rrdbnet-parcel_gf2.pt", map_location=torch.device("cpu"), weights_only=True
            )              
        elif options.train.dataset == "celeba":
            t = torch.load(
                "pretrained-rrdbnet-celeba.pt", map_location=torch.device("cpu"), weights_only=True
            )
        elif options.train.dataset == "imagenet":
            t = torch.load(
                "pretrained-rrdbnet-imagenet.pt", map_location=torch.device("cpu"), weights_only=True
            )
        elif options.train.dataset == "ffhq":
            t = torch.load(
                "pretrained-rrdbnet-ffhq.pt", map_location=torch.device("cpu"), weights_only=True
            )
        else:
            raise ValueError("unknown dataset")
        rrdbnet.load_state_dict(t)
        self.lr_feats = rrdbnet.rrdb

    def forward(self, *args, mode, **kwargs):
        if mode == "loss":
            return self._calculate_loss(*args, **kwargs)
        elif mode == "generate":
            return self._generate_sample(*args, **kwargs)
        else:
            raise ValueError("invalid forward mode")

    def vis_cond(self, cond):
        output_dir = r'D:\github\Populus_SR_GF2_UAV\results\vis'
        # 确保cond是CPU上的张量并转为numpy数组
        cond_np = cond.detach().cpu().numpy()  # [6, 256, 120, 120]
        
        # 创建保存目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取图像尺寸
        height, width = cond_np.shape[2], cond_np.shape[3]
        
        # 为每个batch创建多通道图像
        for b in range(cond_np.shape[0]):
            # 准备输出文件名
            output_path = os.path.join(output_dir, f"batch_{b}_multichannel.tif")
            
            # 创建多通道TIFF文件
            driver = gdal.GetDriverByName('GTiff')
            dataset = driver.Create(
                output_path,
                width,
                height,
                256,  # 256个通道
                gdal.GDT_Float32  # 使用浮点型保存原始值
            )
            
            # 写入每个通道的数据
            for c in range(cond_np.shape[1]):
                band = dataset.GetRasterBand(c+1)  # GDAL波段索引从1开始
                band.WriteArray(cond_np[b, c])
                band.FlushCache()
            
            # 关闭数据集
            dataset = None
        
        exit(-1) # 可视化模式

    def _calculate_loss(self, x, *, cond):
        # print(cond.shape)
        lr_feats = self.lr_feats(cond)
        # self.vis_cond(lr_feats) # 特征图可视化
        cond_scaled = FV.resize(
            cond, (x.shape[2], x.shape[3]), interpolation=FV.InterpolationMode.BICUBIC
        )
        # print(cond_scaled.shape)
        scale = 5
        x = x - cond_scaled
        x = x * scale
        return self.diffusion.normalize(x, cond=(lr_feats, cond_scaled, scale))

    def _generate_sample(self, x, *, cond):
        x = x.view(x.shape[0], self.in_channels, self.image_size_x, self.image_size_y) # 显式重塑输入尺寸
        if self.diffusion.ddim:
            import torch.utils.checkpoint

            lr_feats = torch.utils.checkpoint.checkpoint(self.lr_feats, cond)
        else:
            lr_feats = self.lr_feats(cond)
        cond_scaled = FV.resize(
            cond, (x.shape[2], x.shape[3]), interpolation=FV.InterpolationMode.BICUBIC
        )
        scale = 5
        x = self.diffusion.generate(x, cond=(lr_feats, cond_scaled, scale))
        x = x / scale
        x = x + cond_scaled
        return x

    def set_generate_steps(self, steps):
        def visit(module):
            if isinstance(module, Diffusion):
                module.set_generate_steps(steps)

        self.apply(visit)

    def set_generate_verbose(self, verbose):
        def visit(module):
            if isinstance(module, Diffusion):
                module.set_generate_verbose(verbose)

        self.apply(visit)
