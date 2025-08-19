#!/.conda/envs/dp python
# -*- coding: utf-8 -*-

"""
批量抽取影像指定通道（并行优化版）
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
wanghaoyu191@mails.ucas.ac.cn
"""

import gdal
import ogr
import fnmatch
import os
import sys
import numpy as np
import math
import time
from multiprocessing import Pool, cpu_count
import functools
from scipy.ndimage import zoom

def write_img(out_path, im_proj, im_geotrans, im_data):
    """output img

    Args:
        out_path: Output path
        im_proj: Affine transformation parameters
        im_geotrans: spatial reference
        im_data: Output image data

    """
    # identify data type 
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # calculate number of bands
    if len(im_data.shape) > 2:  
        im_bands, im_height, im_width = im_data.shape
    else:  
        im_bands, (im_height, im_width) = 1, im_data.shape

    # create new img
    driver = gdal.GetDriverByName("GTiff")
    new_dataset = driver.Create(
        out_path, im_width, im_height, im_bands, datatype)
    new_dataset.SetGeoTransform(im_geotrans)
    new_dataset.SetProjection(im_proj)
    if im_bands == 1:
        new_dataset.GetRasterBand(1).WriteArray(im_data.squeeze())
    else:
        for i in range(im_bands):
            new_dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del new_dataset

def read_img(sr_img):
    """read img

    Args:
        sr_img: The full path of the original image

    """
    im_dataset = gdal.Open(sr_img)
    if im_dataset == None:
        print('open sr_img false')
        sys.exit(1)
    im_geotrans = im_dataset.GetGeoTransform()
    im_proj = im_dataset.GetProjection()
    im_width = im_dataset.RasterXSize
    im_height = im_dataset.RasterYSize
    im_data = im_dataset.ReadAsArray(0, 0, im_width, im_height)
    del im_dataset

    return im_data, im_proj, im_geotrans

def process_single_image(img_name, hr_img_path, lr_img_path, output_path, T_value_hr, T_value_lr, binary_band_index, down_scale_factor):
    """处理单个图像的函数，用于并行处理"""
    try:
        hr_img_full_path = os.path.join(hr_img_path, img_name)
        lr_img_full_path = os.path.join(lr_img_path, img_name[:-4]+'x' + str(down_scale_factor) +img_name[-4:])

        data_hr, proj_temp, geotrans_temp = read_img(hr_img_full_path)
        data_lr = read_img(lr_img_full_path)[0]

        # 取指定波段
        data_hr = data_hr[binary_band_index, :, :]
        data_lr = data_lr[binary_band_index, :, :]

        # 将data_lr插值到hr_img尺寸
        data_lr_upsampled = zoom(data_lr, down_scale_factor, order=1)

        # 二值化
        data_hr_binary = np.where(data_hr<T_value_hr, 0, 1)
        data_lr_binary = np.where(data_lr_upsampled<T_value_lr, 0, 1)
                
        # 生成掩膜
        data_mask = np.where(data_hr_binary == data_lr_binary, 1, 0)

        # 输出掩膜
        out_full_path = os.path.join(output_path, img_name)
        write_img(out_full_path, proj_temp, geotrans_temp, data_mask)
        
        return f"Processed {img_name} successfully"
    except Exception as e:
        return f"Error processing {img_name}: {str(e)}"

def main():
    # os.environ['GDAL_DATA'] = r'C:\Users\75198\anaconda3\envs\learn\Lib\site-packages\osgeo\data\gdal' # To prevent ERROR4

    hr_img_path = r'D:\github\Populus_SR_GF2_UAV\data\gupopulus\gupopulus_train_HR'
    lr_img_path = r'D:\github\Populus_SR_GF2_UAV\data\gupopulus\gupopulus_train_LR'
    output_path = r'D:\github\Populus_SR_GF2_UAV\data\gupopulus\gupopulus_train_MASK'
    
    T_value_hr = 60 # 高分图像二值化阈值
    T_value_lr = 70 # 低分图像二值化阈值
    binary_band_index = 0 # 二值化波段

    down_scale_factor = 2

    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)

    # 获取文件列表
    listpic = fnmatch.filter(os.listdir(hr_img_path), '*.png')
    
    # 确定使用的进程数（使用CPU核心数-1，留一个核心给系统）
    num_processes = max(1, cpu_count() - 1)
    
    print(f"Starting processing {len(listpic)} images using {num_processes} processes...")
    start_time = time.time()
    
    # 使用functools.partial创建带有固定参数的函数
    process_func = functools.partial(
        process_single_image,
        hr_img_path=hr_img_path,
        lr_img_path=lr_img_path,
        output_path=output_path,
        T_value_hr=T_value_hr,
        T_value_lr=T_value_lr,
        binary_band_index=binary_band_index,
        down_scale_factor=down_scale_factor
    )
    
    # 创建进程池并并行处理
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_func, listpic)
    
    # 打印处理结果
    for result in results:
        print(result)
    
    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()