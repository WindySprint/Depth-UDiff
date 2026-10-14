import os
import cv2
import kornia
import numpy as np
from PIL import Image
from skimage import img_as_ubyte

import torch

def is_img_file(filename):
    return any(filename.endswith(extension) for extension in
               ['jpg', 'jpeg', 'JPG', 'JPEG', 'png', 'PNG', 'gif', 'bmp', '.BMP'])

def gen_mask_list(root_list, mask_generator, mode='train'):
    imgs_mask = []
    sam_features = []
    print("------------------------------------------------------------------")
    print("Generating masks...")
    for i in range(len(root_list)):
        print(i)
        if(mode == 'train' or mode == 'val'):
            img = Image.open(root_list[i][1])
        else:
            img = Image.open(root_list[i])
        img = np.array(img)
        masks = mask_generator.generate(img)
        for k, mask_data in enumerate(masks):
            mask = mask_data["segmentation"]
            sam_feature = mask_data["features"]
            if k == 0:
                break
        mask = np.array(mask) + 0
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        imgs_mask.append(mask)
        sam_feature = torch.from_numpy(sam_feature).squeeze(0).float()
        sam_features.append(sam_feature)
    imgs_mask = torch.stack(imgs_mask, dim=0)
    sam_features = torch.stack(sam_features, dim=0)

    return imgs_mask, sam_features

def rgb_to_lab(rgb):
    lab = kornia.color.rgb_to_lab(rgb)
    l = lab[:, :1, :, :] / 100.0
    ab = lab[:, 1:, :, :] / 128.0
    lab = torch.cat([l, ab], dim=1)
    return lab

def lab_to_rgb(lab):
    l = lab[:, :1, :, :] * 100
    ab = lab[:, 1:, :, :] * 128.0
    lab = torch.cat([l, ab], dim=1)
    rgb = kornia.color.lab_to_rgb(lab)
    return rgb

def save_image(imgs_out, result_dir, file_names):
    for i in range(imgs_out.shape[0]):
        img_out = torch.clamp(imgs_out[i], 0, 1)
        if img_out.shape[0] == 3:
            img_out = img_out[[2, 1, 0], :, :].permute(1, 2, 0).float().cpu().detach().numpy()
        else:
            img_out = img_out.permute(1, 2, 0).float().cpu().detach().numpy()
        # img_out = img_out.permute(1, 2, 0).float().cpu().detach().numpy()
        img_out = img_as_ubyte(img_out)
        cv2.imwrite(os.path.join(result_dir, file_names[i]), img_out)
