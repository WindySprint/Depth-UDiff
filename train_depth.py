import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler

from models import loss
from utils import dataloader

from models.sde_net import SDENet
from sam import SamAutomaticMaskGenerator, sam_model_registry

torch.cuda.empty_cache()
parser = argparse.ArgumentParser()
# Input Parameters
parser.add_argument('--dnet_name', type=str, default="sde_net")
parser.add_argument('--sam_type', type=str, default="vit_h")
parser.add_argument('--sam_checkpoint', type=str, default="./sam/weights/sam_vit_h_4b8939.pth")
parser.add_argument('--gt_images_path', type=str, default="../../../share_ssd/UIE/datasets/12/train/USOD10K-4000/gt/")
parser.add_argument('--raw_images_path', type=str, default="../../../share_ssd/UIE/datasets/12/train/USOD10K-4000/input/")
parser.add_argument('--n_feats', type=int, default=16)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--grad_clip_norm', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--train_batch_size', type=int, default=4)
parser.add_argument('--val_batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=20)
parser.add_argument('--checkpoint_path', type=str, default="checkpoints/")
config = parser.parse_args()

def train(config):
    ######### Model ###########
    d_net = SDENet(config.n_feats).cuda()

    sam_name = config.sam_type
    sam = sam_model_registry[sam_name](checkpoint=config.sam_checkpoint).cuda()
    sam.eval()
    mask_generator = SamAutomaticMaskGenerator(sam)

    if not os.path.exists(os.path.join(config.checkpoint_path, config.dnet_name)):
        os.mkdir(os.path.join(config.checkpoint_path, config.dnet_name))

    criterion_mse = loss.MSE_Loss()

    train_dataset = dataloader.d_train_val_loader(config.gt_images_path, config.raw_images_path, mask_generator, 'train')
    val_dataset = dataloader.d_train_val_loader(config.gt_images_path, config.raw_images_path, mask_generator, 'val')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False,
                                             num_workers=config.num_workers, pin_memory=True)

    ######### Adam optimizer ###########
    optimizer = optim.Adam(d_net.parameters(), lr=config.lr)
    ######### Scheduler ###########
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.num_epochs - warmup_epochs,
                                                            eta_min=config.lr)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()

    d_net.train()

    # Record best index and corresponding epoch
    best_psnr = 0
    best_epoch = 0

    for epoch in range(1, config.num_epochs + 1):
        epoch_start_time = time.time()
        # Record train loss and validation index
        train_loss = []
        val_psnr = []
        print("*" * 30 + "The %i epoch" % epoch + "*" * 30 + '\n')

        for _, (img_gt, img_raw, img_mask, sam_features) in enumerate(tqdm(train_loader)):
            img_gt = img_gt.cuda()
            img_raw = img_raw.cuda()
            img_mask = img_mask.cuda()
            sam_features = sam_features.cuda()

            try:
                img_d = d_net(img_raw, img_mask, sam_features)
                mse_loss = criterion_mse(img_gt, img_d)
                sum_loss = mse_loss

                train_loss.append(sum_loss.item())
                optimizer.zero_grad()
                sum_loss.backward()
                torch.nn.utils.clip_grad_norm_(d_net.parameters(), config.grad_clip_norm)
                optimizer.step()

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(e)
                    torch.cuda.empty_cache()
                else:
                    raise e

        with open(os.path.join(config.checkpoint_path, config.dnet_name, "loss.log"), "a+", encoding="utf-8") as f:
            s = "The %i Epoch mean_loss is :%f" % (epoch, np.mean(train_loss)) + "\n"
            f.write(s)

        # Validation Stage
        with torch.no_grad():
            for _, (img_gt, img_raw, img_mask, sam_features) in enumerate(val_loader):
                img_gt = img_gt.cuda()
                img_raw = img_raw.cuda()
                img_mask = img_mask.cuda()
                sam_features = sam_features.cuda()

                img_d = d_net(img_raw, img_mask, sam_features)

                psnr = loss.torchPSNR(img_gt, img_d)
                val_psnr.append(psnr.item())

        val_psnr = np.mean(np.array(val_psnr))

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_epoch = epoch
            torch.save({'state_dict': d_net.state_dict()},
                       os.path.join(config.checkpoint_path, config.dnet_name, "model_best.pth"))

        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" %
              (epoch, val_psnr, best_epoch, best_psnr))

        with open(os.path.join(config.checkpoint_path, config.dnet_name, "val_PSNR.log"), "a+", encoding="utf-8") as f:
            f.write("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" %
                    (epoch, val_psnr, best_epoch, best_psnr) + "\n")

        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format
              (epoch, time.time() - epoch_start_time, np.mean(train_loss), scheduler.get_lr()[0]))

        scheduler.step()
        torch.save({'state_dict': d_net.state_dict()},
                   os.path.join(config.checkpoint_path, config.dnet_name, "model_latest.pth"))


if __name__ == "__main__":
    start_time = time.time()
    train(config)
    e = time.time()
    print("train_time:" + str(time.time() - start_time))
