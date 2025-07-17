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

def write_img(out_path, im_proj, im_geotrans, im_data, datatype):
    """output img

    Args:
        out_path: Output path
        im_proj: Affine transformation parameters
        im_geotrans: spatial reference
        im_data: Output image data

    """
    # calculate number of bands
    if len(im_data.shape) > 2:  
        im_bands, im_height, im_width = im_data.shape
    else:  
        im_bands, (im_height, im_width) = 1, im_data.shape

    # create new img
    driver = gdal.GetDriverByName("GTiff")
    new_dataset = driver.Create(
        out_path, im_width, im_height, im_bands, datatype)
    # new_dataset.SetGeoTransform(im_geotrans)
    # new_dataset.SetProjection(im_proj)
    if im_bands == 1:
        new_dataset.GetRasterBand(1).WriteArray(im_data.squeeze())
    else:
        for i in range(im_bands):
            new_dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del new_dataset

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
        elif options.train.dataset == "gupopulus" or options.train.dataset == "gupopulus_mask" or options.train.dataset == "gupopulus_D":
            t = torch.load(
                "pretrained-rrdbnet-gupopulus" + "-x" + str(options.model.sr_factor)+".pt", map_location=torch.device("cpu"), weights_only=True
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
        elif mode == "loss-mask":
            return self._calculate_loss_mask(*args, **kwargs)
        elif mode == "generate":
            return self._generate_sample(*args, **kwargs)
        else:
            raise ValueError("invalid forward mode")

    def vis_cond(self, lr_feats):
        
        output_path = r'D:\github_respository\Populus_SR_GF2_UAV\results\20250409-152034-gupopulus_250409\vis'
        # 确保cond是CPU上的张量并转为numpy数组
        lr_feats_np = lr_feats[0].detach().cpu().numpy()
        # 创建保存目录
        os.makedirs(output_path, exist_ok=True)
        output_full_path = output_path + '/' + 'lr_feats.tif'
        write_img(output_full_path, None, None, lr_feats_np, datatype=gdal.GDT_Float32)


    def vis_tstep(self, x, scale, cond_scaled):
        x_copy = x.clone()
        x_copy = x_copy / scale
        x_copy = x_copy + cond_scaled
        x_copy = (x_copy.clamp(0, 1) * 255).round()
        x_copy = x_copy.detach().cpu().numpy().astype(np.uint8)

        output_path = r'D:\github_respository\Populus_SR_GF2_UAV\results\20250409-152034-gupopulus_250409\vis'
        os.makedirs(output_path, exist_ok=True)

        for i in range(x_copy.shape[0]):
            output_data = x_copy[i][0]
            output_full_path = output_path + '/' + 'step_' + str(i) + '.tif'
            write_img(output_full_path, None, None, output_data, datatype=gdal.GDT_Byte)
            

    def _calculate_loss(self, x, *, cond):
        lr_feats = self.lr_feats(cond)
        cond_scaled = FV.resize(
            cond, (x.shape[2], x.shape[3]), interpolation=FV.InterpolationMode.BICUBIC
        )
        scale = 5
        x = x - cond_scaled
        x = x * scale
        return self.diffusion.normalize(x, cond=(lr_feats, cond_scaled, scale))
    
    def _calculate_loss_mask(self, x, *, cond, mask):
        lr_feats = self.lr_feats(cond)
        cond_scaled = FV.resize(
            cond, (x.shape[2], x.shape[3]), interpolation=FV.InterpolationMode.BICUBIC
        )
        scale = 5
        x = x - cond_scaled
        x = x * scale
        return self.diffusion.normalize_mask(x, cond=(lr_feats, cond_scaled, scale), mask=mask)        

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
        
        # # 可视化
        # self.vis_cond(lr_feats)
        # self.vis_tstep(x, scale, cond_scaled)

        if self.diffusion.use_ode:
            x = x[-1]
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
