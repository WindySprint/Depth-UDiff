import os
import torch
import torch.utils.data as data
import numpy as np
import random
import cv2
import kornia

from utils import data_util

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif', 'bmp'])

def d_train_val_list(gt_images_path, ori_images_path):
    image_list_index = os.listdir(ori_images_path)
    all_length = len(image_list_index)
    image_list_index = random.sample(image_list_index, all_length)

    image_dataset = []
    for i in image_list_index:  # Add paths and combine them
        image_dataset.append((gt_images_path + i, ori_images_path + i))

    train_list = image_dataset[:int(all_length*0.9)]
    val_list = image_dataset[int(all_length*0.9):]

    return train_list, val_list

def train_val_list(gt_images_path, ori_images_path, d_images_path):
    image_list_index = os.listdir(ori_images_path)
    all_length = len(image_list_index)
    image_list_index = random.sample(image_list_index, all_length)

    image_dataset = []
    for i in image_list_index:  # Add paths and combine them
        image_dataset.append((gt_images_path + i, ori_images_path + i, d_images_path + i))

    train_list = image_dataset[:int(all_length*0.9)]
    val_list = image_dataset[int(all_length*0.9):]

    return train_list, val_list

class d_train_val_loader(data.Dataset):

    def __init__(self, gt_images_path, ori_images_path, mask_generator, mode='train'):
        self.train_list, self.val_list = d_train_val_list(gt_images_path, ori_images_path)
        self.mode = mode
        if self.mode == 'train':
            self.data_list = self.train_list
            print("Total training examples:", len(self.train_list))
        else:
            self.data_list = self.val_list
            print("Total validation examples:", len(self.val_list))
        self.imgs_mask, self.sam_features = data_util.gen_mask_list(self.data_list, mask_generator, self.mode)

    def __getitem__(self, index):

        data_gt_path, data_raw_path = self.data_list[index]

        data_gt = cv2.imread(data_gt_path, cv2.IMREAD_GRAYSCALE)
        data_raw = cv2.imread(data_raw_path)

        data_gt = np.asarray(data_gt) / 255.0
        data_raw = np.asarray(data_raw) / 255.0
        data_mask = np.asarray(self.imgs_mask[index])
        data_feature = np.asarray(self.sam_features[index])

        data_gt = torch.from_numpy(data_gt).unsqueeze(2).float()
        data_raw = torch.from_numpy(data_raw).float()
        data_mask = torch.from_numpy(data_mask).float()
        data_feature = torch.from_numpy(data_feature).float()

        return data_gt.permute(2, 0, 1), data_raw.permute(2, 0, 1), data_mask, data_feature

    def __len__(self):
        return len(self.data_list)

class train_val_loader(data.Dataset):

    def __init__(self, gt_images_path, ori_images_path, d_images_path, mode='train'):
        self.train_list, self.val_list = train_val_list(gt_images_path, ori_images_path, d_images_path)
        self.mode = mode
        if self.mode == 'train':
            self.data_list = self.train_list
            print("Total training examples:", len(self.train_list))
        else:
            self.data_list = self.val_list
            print("Total validation examples:", len(self.val_list))

    def __getitem__(self, index):

        data_gt_path, data_raw_path, data_d_path = self.data_list[index]

        data_gt = cv2.imread(data_gt_path)
        data_raw = cv2.imread(data_raw_path)
        data_d = cv2.imread(data_d_path, cv2.IMREAD_GRAYSCALE)

        data_gt = cv2.cvtColor(data_gt, cv2.COLOR_BGR2RGB)
        data_raw = cv2.cvtColor(data_raw, cv2.COLOR_BGR2RGB)

        data_gt = np.asarray(data_gt) / 255.0
        data_raw = np.asarray(data_raw) / 255.0
        data_d = np.asarray(data_d) / 255.0

        data_gt = torch.from_numpy(data_gt).float()
        data_raw = torch.from_numpy(data_raw).float()
        data_d = torch.from_numpy(data_d).unsqueeze(2).float()

        return data_gt.permute(2, 0, 1), data_raw.permute(2, 0, 1), data_d.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)

class test_loader(data.Dataset):
    def __init__(self, ori_images_path, mask_generator):
        super(test_loader, self).__init__()

        image_list_index = sorted(os.listdir(ori_images_path))
        self.image_dataset = [os.path.join(ori_images_path, x) for x in image_list_index if is_image_file(x)]
        self.all_length = len(self.image_dataset)
        self.imgs_mask, self.sam_features = data_util.gen_mask_list(self.image_dataset, mask_generator, 'test')

    def __len__(self):
        return self.all_length

    def __getitem__(self, index):

        data_raw_path = self.image_dataset[index]
        filename = data_raw_path.split('/')[-1]
        data_raw = cv2.imread(data_raw_path)
        data_raw = cv2.cvtColor(data_raw, cv2.COLOR_BGR2RGB)

        data_raw = np.asarray(data_raw) / 255.0
        data_mask = np.asarray(self.imgs_mask[index])
        data_feature = np.asarray(self.sam_features[index])

        data_raw = torch.from_numpy(data_raw).float()
        data_mask = torch.from_numpy(data_mask).float()
        data_feature = torch.from_numpy(data_feature).float()

        return data_raw.permute(2, 0, 1), data_mask, data_feature, filename

class test_loader_withdepth(data.Dataset):
    def __init__(self, ori_images_path, d_images_path):
        super(test_loader_withdepth, self).__init__()

        image_list_index = sorted(os.listdir(ori_images_path))
        depth_list_index = sorted(os.listdir(d_images_path))
        self.image_dataset = [os.path.join(ori_images_path, x) for x in image_list_index if is_image_file(x)]
        self.depth_dataset = [os.path.join(d_images_path, x) for x in depth_list_index if is_image_file(x)]
        self.all_length = len(self.image_dataset)

    def __len__(self):
        return self.all_length

    def __getitem__(self, index):

        data_raw_path = self.image_dataset[index]
        data_d_path = self.depth_dataset[index]
        filename = data_raw_path.split('/')[-1]
        data_raw = cv2.imread(data_raw_path)
        data_raw = cv2.cvtColor(data_raw, cv2.COLOR_BGR2RGB)
        data_d = cv2.imread(data_d_path, cv2.IMREAD_GRAYSCALE)

        data_raw = np.asarray(data_raw) / 255.0
        data_d = np.asarray(data_d) / 255.0

        data_raw = torch.from_numpy(data_raw).float()
        data_d = torch.from_numpy(data_d).unsqueeze(2).float()

        return data_raw.permute(2, 0, 1), data_d.permute(2, 0, 1), filename